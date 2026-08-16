#include "wavefront_renderer.h"

#include "cuda_error.h"
#include "device_buffer.h"
#include "packed_transport_core.h"
#include "workspace_memory.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace cuda_backend {
namespace {

constexpr std::uint32_t kDefaultBatchSize = 256u * 1024u;

struct alignas(16) DeviceRenderCounters {
    unsigned long long traversal_steps = 0;
    unsigned long long shadow_rays = 0;
    unsigned long long clamped_samples = 0;
    unsigned long long invalid_samples = 0;
    unsigned long long status_counts[8]{};
    unsigned long long depth_steps = 0;
    unsigned long long active_path_steps = 0;
};

static_assert(std::is_trivially_copyable_v<DeviceRenderCounters>);

class CudaEvent {
public:
    CudaEvent() {
        RT_CUDA_CHECK(cudaEventCreate(&m_event));
    }

    ~CudaEvent() {
        if (m_event != nullptr) {
            cudaEventDestroy(m_event);
        }
    }

    CudaEvent(const CudaEvent &) = delete;
    CudaEvent &operator=(const CudaEvent &) = delete;

    operator cudaEvent_t() const noexcept {
        return m_event;
    }

private:
    cudaEvent_t m_event = nullptr;
};

RT_HOST_DEVICE RT_FORCE_INLINE float luminance(Float3 value) {
    return 0.2126f * value.x + 0.7152f * value.y + 0.0722f * value.z;
}

__global__ void render_paths_kernel(
    DeviceSceneView scene, PackedTransportSettings transport,
    std::uint32_t width, std::uint32_t height,
    std::uint32_t pixel_offset, std::uint32_t path_count,
    std::uint32_t sample_index, std::uint32_t seed,
    PackedPathState *states, float sample_clamp, CudaFilmPixel *film,
    DeviceRenderCounters *counters) {
    extern __shared__ unsigned char dynamic_shared[];
    unsigned *queues = reinterpret_cast<unsigned *>(dynamic_shared);

    __shared__ std::uint32_t counts[2];
    __shared__ unsigned long long status_partials[8];
    __shared__ unsigned long long traversal_partial;
    __shared__ unsigned long long shadow_partial;
    __shared__ unsigned long long invalid_partial;
    __shared__ unsigned long long clamped_partial;
    __shared__ unsigned long long depth_steps_partial;
    __shared__ unsigned long long active_path_partial;

    const unsigned thread = threadIdx.x;
    if (thread < 8) {
        status_partials[thread] = 0;
    }
    if (thread == 0) {
        counts[0] = 0;
        counts[1] = 0;
        traversal_partial = 0;
        shadow_partial = 0;
        invalid_partial = 0;
        clamped_partial = 0;
        depth_steps_partial = 0;
        active_path_partial = 0;
    }
    __syncthreads();

    const std::uint32_t local_index =
        blockIdx.x * blockDim.x + threadIdx.x;
    const bool valid = local_index < path_count;
    if (valid) {
        const std::uint32_t pixel_index = pixel_offset + local_index;
        const std::uint32_t pixel_x = pixel_index % width;
        const std::uint32_t pixel_y = pixel_index / width;
        const std::uint32_t sample_seed =
            packed_transport::packed_camera_sample_seed(
                seed, pixel_index, sample_index);
        RNG rng(sample_seed);
        const PackedRay ray =
            packed_transport::generate_packed_camera_ray_core(
                scene.scene.camera, pixel_x, pixel_y, width, height, rng);
        packed_transport::initialize_packed_path_state(
            scene.scene, ray, transport, rng.state, pixel_index,
            sample_index, states[local_index]);
        if (states[local_index].active()) {
            const std::uint32_t slot = atomicAdd(&counts[0], 1u);
            queues[slot] = local_index;
        }
    }
    __syncthreads();

    std::uint32_t active = counts[0];
    std::uint32_t parity = 0;
    while (active != 0u) {
        const std::uint32_t next_parity = parity ^ 1u;
        const unsigned *source_queue = queues + parity * blockDim.x;
        unsigned *next_queue = queues + next_parity * blockDim.x;
        if (thread < active) {
            const std::uint32_t path_index = source_queue[thread];
            PackedPathState state = states[path_index];
            packed_transport::advance_packed_path_core(scene.scene, transport,
                                                       state);
            states[path_index] = state;
            if (state.active()) {
                const std::uint32_t output_index =
                    atomicAdd(&counts[next_parity], 1u);
                next_queue[output_index] = path_index;
            }
        }
        __syncthreads();
        active = counts[next_parity];
        __syncthreads();
        if (thread == 0) {
            depth_steps_partial += 1u;
            active_path_partial += active;
            counts[next_parity] = 0;
        }
        __syncthreads();
        parity = next_parity;
    }
    __syncthreads();

    // Every path slot belongs to exactly one block, so the film update needs
    // no atomic. Counter updates are reduced per block first.
    if (valid) {
        const PackedPathState &state = states[local_index];
        Float3 radiance = state.radiance;
        std::uint32_t status_index =
            static_cast<std::uint32_t>(state.status);
        if (status_index >= 8) {
            status_index = static_cast<std::uint32_t>(
                PackedTransportStatus::InvalidInput);
        }

        atomicAdd(&status_partials[status_index], 1ull);
        atomicAdd(&traversal_partial,
                  static_cast<unsigned long long>(state.traversal_steps));
        atomicAdd(&shadow_partial,
                  static_cast<unsigned long long>(state.shadow_rays));

        if (state.status != PackedTransportStatus::Success ||
            !packed_transport::math::finite(radiance)) {
            radiance = {};
            atomicAdd(&invalid_partial, 1ull);
        } else if (sample_clamp > 0.0f) {
            const float sample_luminance = luminance(radiance);
            if (sample_luminance > sample_clamp) {
                const float scale = sample_clamp / sample_luminance;
                radiance =
                    packed_transport::math::multiply(radiance, scale);
                atomicAdd(&clamped_partial, 1ull);
            }
        }

        CudaFilmPixel &pixel = film[state.pixel_index];
        pixel.radiance =
            packed_transport::math::add(pixel.radiance, radiance);
        ++pixel.sample_count;
    }

    __syncthreads();
    if (thread == 0) {
        for (unsigned index = 0; index < 8; ++index) {
            atomicAdd(&counters->status_counts[index],
                      status_partials[index]);
        }
        atomicAdd(&counters->traversal_steps, traversal_partial);
        atomicAdd(&counters->shadow_rays, shadow_partial);
        atomicAdd(&counters->invalid_samples, invalid_partial);
        atomicAdd(&counters->clamped_samples, clamped_partial);
        atomicAdd(&counters->depth_steps, depth_steps_partial);
        atomicAdd(&counters->active_path_steps, active_path_partial);
    }
}

void validate_settings(const CudaRenderSettings &settings) {
    if (settings.width == 0 || settings.height == 0 ||
        settings.samples_per_pixel == 0) {
        throw std::invalid_argument(
            "CUDA render dimensions and sample count must be positive");
    }
    if (settings.transport.max_depth == 0 ||
        !valid_integrator_policy(settings.transport.policy)) {
        throw std::invalid_argument("invalid CUDA transport settings");
    }
    if (settings.block_size == 0 || settings.block_size > 1024) {
        throw std::invalid_argument("invalid CUDA render block size");
    }
    if (!(settings.sample_clamp >= 0.0f)) {
        throw std::invalid_argument("CUDA sample clamp must be non-negative");
    }
    const std::uint64_t pixel_count =
        static_cast<std::uint64_t>(settings.width) * settings.height;
    if (pixel_count > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("CUDA film exceeds uint32 capacity");
    }
}

std::uint32_t choose_batch_size(const CudaRenderSettings &settings,
                                std::uint32_t pixel_count) {
    const std::uint32_t requested = settings.batch_size == 0
                                        ? kDefaultBatchSize
                                        : settings.batch_size;
    return std::max(1u, std::min(requested, pixel_count));
}

std::size_t workspace_bytes(std::uint32_t pixel_count,
                            std::uint32_t batch_size) {
    return static_cast<std::size_t>(pixel_count) * sizeof(CudaFilmPixel) +
           static_cast<std::size_t>(batch_size) * sizeof(PackedPathState) +
           sizeof(DeviceRenderCounters);
}

} // namespace

struct CudaRenderWorkspace::Impl {
    DeviceBuffer<CudaFilmPixel> film;
    DeviceBuffer<PackedPathState> states;
    DeviceBuffer<DeviceRenderCounters> counters;
    std::uint64_t generation = 0;

    std::size_t allocated_bytes() const noexcept {
        return film.bytes() + states.bytes() + counters.bytes();
    }

    void ensure_capacity(std::uint32_t pixel_count,
                         std::uint32_t batch_size) {
        const std::uint32_t pixel_capacity =
            std::max(pixel_count, film.size());
        const std::uint32_t path_capacity =
            std::max(batch_size, states.size());
        const std::size_t required =
            workspace_bytes(pixel_capacity, path_capacity);
        const std::size_t current = allocated_bytes();
        if (required > current) {
            ensure_cuda_workspace_fits(required, current,
                                       "CUDA render workspace");
        }

        bool changed = false;
        changed |= film.ensure_capacity_discard(pixel_count);
        changed |= states.ensure_capacity_discard(batch_size);
        changed |= counters.ensure_capacity_discard(1);
        if (changed) {
            ++generation;
        }
    }

    CudaWorkspaceInfo info() const noexcept {
        CudaWorkspaceInfo result;
        result.bytes = allocated_bytes();
        result.generation = generation;
        result.pixel_capacity = film.size();
        result.path_capacity = states.size();
        result.film_address =
            reinterpret_cast<std::uintptr_t>(film.data());
        result.path_address =
            reinterpret_cast<std::uintptr_t>(states.data());
        return result;
    }
};

CudaRenderWorkspace::CudaRenderWorkspace()
    : m_impl(std::make_unique<Impl>()) {
}

CudaRenderWorkspace::~CudaRenderWorkspace() = default;
CudaRenderWorkspace::CudaRenderWorkspace(CudaRenderWorkspace &&) noexcept =
    default;
CudaRenderWorkspace &CudaRenderWorkspace::operator=(
    CudaRenderWorkspace &&) noexcept = default;

CudaWorkspaceInfo CudaRenderWorkspace::info() const noexcept {
    return m_impl ? m_impl->info() : CudaWorkspaceInfo{};
}

CudaRenderOutput render_wavefront_cuda(
    DeviceSceneView scene, const CudaRenderSettings &settings,
    CudaRenderWorkspace &workspace, const std::atomic<bool> *cancel) {
    validate_settings(settings);
    if (scene.scene.aggregates.count == 0) {
        throw std::invalid_argument("CUDA render scene is empty");
    }

    const std::uint32_t pixel_count = settings.width * settings.height;
    const std::uint32_t batch_size =
        choose_batch_size(settings, pixel_count);
    workspace.m_impl->ensure_capacity(pixel_count, batch_size);
    CudaRenderWorkspace::Impl &buffers = *workspace.m_impl;
    RT_CUDA_CHECK(cudaMemset(buffers.film.data(), 0,
                             static_cast<std::size_t>(pixel_count) *
                                 sizeof(CudaFilmPixel)));
    RT_CUDA_CHECK(cudaMemset(buffers.counters.data(), 0,
                             sizeof(DeviceRenderCounters)));

    CudaEvent begin;
    CudaEvent end;
    RT_CUDA_CHECK(cudaEventRecord(begin));
    std::uint32_t batch_count = 0;
    bool cancelled = false;
    const std::size_t shared_bytes =
        static_cast<std::size_t>(2) * settings.block_size *
        sizeof(std::uint32_t);
    for (std::uint32_t sample = 0;
         sample < settings.samples_per_pixel && !cancelled; ++sample) {
        for (std::uint32_t offset = 0; offset < pixel_count;
             offset += batch_size) {
            if (cancel != nullptr && cancel->load()) {
                cancelled = true;
                break;
            }
            ++batch_count;
            const std::uint32_t count =
                std::min(batch_size, pixel_count - offset);
            const std::uint32_t grid =
                (count + settings.block_size - 1) / settings.block_size;
            render_paths_kernel<<<grid, settings.block_size, shared_bytes>>>(
                scene, settings.transport, settings.width, settings.height,
                offset, count, sample, settings.seed, buffers.states.data(),
                static_cast<float>(settings.sample_clamp),
                buffers.film.data(), buffers.counters.data());
            RT_CUDA_CHECK(cudaGetLastError());
            // One kernel owns a complete batch. The default stream orders all
            // batches, so the final event sync is the only required host
            // synchronization point.
        }
    }
    RT_CUDA_CHECK(cudaEventRecord(end));
    RT_CUDA_CHECK(cudaEventSynchronize(end));

    CudaRenderOutput output;
    buffers.film.download_prefix(output.film, pixel_count);
    std::vector<DeviceRenderCounters> host_counters;
    buffers.counters.download_prefix(host_counters, 1);
    const DeviceRenderCounters &counter = host_counters.front();
    RT_CUDA_CHECK(cudaEventElapsedTime(&output.stats.milliseconds, begin, end));
    output.stats.traversal_steps = counter.traversal_steps;
    output.stats.shadow_rays = counter.shadow_rays;
    output.stats.clamped_samples = counter.clamped_samples;
    output.stats.invalid_samples = counter.invalid_samples;
    output.stats.batch_size = batch_size;
    output.stats.batch_count = batch_count;
    output.stats.advance_launches = counter.depth_steps;
    output.stats.active_path_steps = counter.active_path_steps;
    const CudaWorkspaceInfo workspace_info = workspace.info();
    output.stats.workspace_bytes = workspace_info.bytes;
    output.stats.workspace_generation = workspace_info.generation;
    output.stats.workspace_pixel_capacity = workspace_info.pixel_capacity;
    output.stats.workspace_path_capacity = workspace_info.path_capacity;
    output.stats.cancelled = cancelled;
    for (std::size_t index = 0; index < output.stats.status_counts.size();
         ++index) {
        output.stats.status_counts[index] = counter.status_counts[index];
        output.stats.sample_count += counter.status_counts[index];
    }
    return output;
}

CudaRenderOutput render_wavefront_cuda(
    DeviceSceneView scene, const CudaRenderSettings &settings,
    const std::atomic<bool> *cancel) {
    CudaRenderWorkspace workspace;
    return render_wavefront_cuda(scene, settings, workspace, cancel);
}

} // namespace cuda_backend

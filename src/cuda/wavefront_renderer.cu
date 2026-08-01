#include "wavefront_renderer.h"

#include "cuda_error.h"
#include "device_buffer.h"
#include "render_data/packed_transport_core.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace cuda_backend {
namespace {

constexpr std::uint32_t kDefaultBatchSize = 256u * 1024u;
constexpr std::size_t kMemoryReserve = 64ull * 1024ull * 1024ull;

struct alignas(16) DeviceRenderCounters {
    unsigned long long traversal_steps = 0;
    unsigned long long shadow_rays = 0;
    unsigned long long clamped_samples = 0;
    unsigned long long invalid_samples = 0;
    unsigned long long status_counts[8]{};
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

__global__ void initialize_paths_kernel(
    DeviceSceneView scene, PackedTransportSettings transport,
    std::uint32_t width, std::uint32_t height,
    std::uint32_t pixel_offset, std::uint32_t path_count,
    std::uint32_t sample_index, std::uint32_t seed,
    PackedPathState *states, std::uint32_t *active_indices) {
    const std::uint32_t local_index =
        blockIdx.x * blockDim.x + threadIdx.x;
    if (local_index >= path_count) {
        return;
    }
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
        scene.scene, ray, transport, rng.state, pixel_index, sample_index,
        states[local_index]);
    active_indices[local_index] = local_index;
}

__global__ void advance_paths_kernel(
    DeviceSceneView scene, PackedTransportSettings transport,
    PackedPathState *states, const std::uint32_t *active_indices,
    std::uint32_t active_count, std::uint32_t *next_indices,
    std::uint32_t *next_count) {
    const std::uint32_t queue_index =
        blockIdx.x * blockDim.x + threadIdx.x;
    if (queue_index >= active_count) {
        return;
    }
    const std::uint32_t path_index = active_indices[queue_index];
    PackedPathState state = states[path_index];
    packed_transport::advance_packed_path_core(scene.scene, transport,
                                               state);
    states[path_index] = state;
    if (state.active()) {
        const std::uint32_t output_index = atomicAdd(next_count, 1u);
        next_indices[output_index] = path_index;
    }
}

__global__ void accumulate_paths_kernel(
    const PackedPathState *states, std::uint32_t path_count,
    float sample_clamp, CudaFilmPixel *film,
    DeviceRenderCounters *counters) {
    const std::uint32_t path_index =
        blockIdx.x * blockDim.x + threadIdx.x;
    if (path_index >= path_count) {
        return;
    }
    const PackedPathState &state = states[path_index];
    Float3 radiance = state.radiance;
    std::uint32_t status_index =
        static_cast<std::uint32_t>(state.status);
    if (status_index >= 8) {
        status_index =
            static_cast<std::uint32_t>(PackedTransportStatus::InvalidInput);
    }
    atomicAdd(&counters->status_counts[status_index], 1ull);
    atomicAdd(&counters->traversal_steps,
              static_cast<unsigned long long>(state.traversal_steps));
    atomicAdd(&counters->shadow_rays,
              static_cast<unsigned long long>(state.shadow_rays));

    if (state.status != PackedTransportStatus::Success ||
        !packed_transport::math::finite(radiance)) {
        radiance = {};
        atomicAdd(&counters->invalid_samples, 1ull);
    } else if (sample_clamp > 0.0f) {
        const float sample_luminance = luminance(radiance);
        if (sample_luminance > sample_clamp) {
            const float scale = sample_clamp / sample_luminance;
            radiance = packed_transport::math::multiply(radiance, scale);
            atomicAdd(&counters->clamped_samples, 1ull);
        }
    }

    CudaFilmPixel &pixel = film[state.pixel_index];
    pixel.radiance =
        packed_transport::math::add(pixel.radiance, radiance);
    ++pixel.sample_count;
}

void validate_settings(const CudaRenderSettings &settings) {
    if (settings.width == 0 || settings.height == 0 ||
        settings.samples_per_pixel == 0) {
        throw std::invalid_argument(
            "CUDA render dimensions and sample count must be positive");
    }
    if (settings.transport.max_depth == 0 ||
        static_cast<std::uint32_t>(settings.transport.integrator) >
            static_cast<std::uint32_t>(PackedIntegratorType::MISPath)) {
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
           static_cast<std::size_t>(batch_size) *
               2 * sizeof(std::uint32_t) +
           sizeof(std::uint32_t) + sizeof(DeviceRenderCounters);
}

void ensure_workspace_fits(std::size_t bytes) {
    const DeviceMemoryInfo memory = query_device_memory();
    const std::size_t usable = memory.free_bytes > kMemoryReserve
                                   ? memory.free_bytes - kMemoryReserve
                                   : 0;
    if (bytes <= usable) {
        return;
    }
    std::ostringstream message;
    message << "CUDA render workspace requires " << bytes
            << " bytes, but only " << memory.free_bytes
            << " device bytes are free";
    throw std::runtime_error(message.str());
}

} // namespace

CudaRenderOutput render_wavefront_cuda(
    DeviceSceneView scene, const CudaRenderSettings &settings,
    const std::atomic<bool> *cancel) {
    validate_settings(settings);
    if (scene.scene.aggregates.count == 0) {
        throw std::invalid_argument("CUDA render scene is empty");
    }

    const std::uint32_t pixel_count = settings.width * settings.height;
    const std::uint32_t batch_size =
        choose_batch_size(settings, pixel_count);
    const std::size_t required_workspace =
        workspace_bytes(pixel_count, batch_size);
    ensure_workspace_fits(required_workspace);

    DeviceBuffer<CudaFilmPixel> film;
    DeviceBuffer<PackedPathState> states;
    DeviceBuffer<std::uint32_t> active_indices;
    DeviceBuffer<std::uint32_t> next_indices;
    DeviceBuffer<std::uint32_t> next_count;
    DeviceBuffer<DeviceRenderCounters> counters;
    film.allocate(pixel_count);
    states.allocate(batch_size);
    active_indices.allocate(batch_size);
    next_indices.allocate(batch_size);
    next_count.allocate(1);
    counters.allocate(1);
    RT_CUDA_CHECK(cudaMemset(film.data(), 0, film.bytes()));
    RT_CUDA_CHECK(cudaMemset(counters.data(), 0, counters.bytes()));

    CudaEvent begin;
    CudaEvent end;
    RT_CUDA_CHECK(cudaEventRecord(begin));
    std::uint32_t batch_count = 0;
    bool cancelled = false;
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
            initialize_paths_kernel<<<grid, settings.block_size>>>(
                scene, settings.transport, settings.width, settings.height,
                offset, count, sample, settings.seed, states.data(),
                active_indices.data());
            RT_CUDA_CHECK(cudaGetLastError());

            std::uint32_t active_paths = count;
            for (std::uint32_t depth = 0;
                 depth < settings.transport.max_depth && active_paths != 0;
                 ++depth) {
                RT_CUDA_CHECK(cudaMemset(next_count.data(), 0,
                                         sizeof(std::uint32_t)));
                const std::uint32_t active_grid =
                    (active_paths + settings.block_size - 1) /
                    settings.block_size;
                advance_paths_kernel<<<active_grid, settings.block_size>>>(
                    scene, settings.transport, states.data(),
                    active_indices.data(), active_paths,
                    next_indices.data(), next_count.data());
                RT_CUDA_CHECK(cudaGetLastError());
                RT_CUDA_CHECK(cudaMemcpy(&active_paths, next_count.data(),
                                         sizeof(active_paths),
                                         cudaMemcpyDeviceToHost));
                active_indices.swap(next_indices);
            }

            accumulate_paths_kernel<<<grid, settings.block_size>>>(
                states.data(), count, settings.sample_clamp, film.data(),
                counters.data());
            RT_CUDA_CHECK(cudaGetLastError());
        }
    }
    RT_CUDA_CHECK(cudaEventRecord(end));
    RT_CUDA_CHECK(cudaEventSynchronize(end));

    CudaRenderOutput output;
    film.download(output.film);
    std::vector<DeviceRenderCounters> host_counters;
    counters.download(host_counters);
    const DeviceRenderCounters &counter = host_counters.front();
    RT_CUDA_CHECK(cudaEventElapsedTime(&output.stats.milliseconds, begin, end));
    output.stats.traversal_steps = counter.traversal_steps;
    output.stats.shadow_rays = counter.shadow_rays;
    output.stats.clamped_samples = counter.clamped_samples;
    output.stats.invalid_samples = counter.invalid_samples;
    output.stats.batch_size = batch_size;
    output.stats.batch_count = batch_count;
    output.stats.workspace_bytes = required_workspace;
    output.stats.cancelled = cancelled;
    for (std::size_t index = 0; index < output.stats.status_counts.size();
         ++index) {
        output.stats.status_counts[index] = counter.status_counts[index];
        output.stats.sample_count += counter.status_counts[index];
    }
    return output;
}

} // namespace cuda_backend

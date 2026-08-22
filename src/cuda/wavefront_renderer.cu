#include "wavefront_renderer.h"

#include "cuda_error.h"
#include "device_buffer.h"
#include "packed_transport_core.h"
#include "workspace_memory.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace cuda_backend {
namespace {

constexpr std::uint32_t kDefaultBatchSize = 256u * 1024u;
constexpr std::uint32_t kBlockSizeFallback = 128u;
constexpr std::uint32_t kAutotuneTargetWidth = 256u;
constexpr std::uint32_t kAutotuneMeasurementRuns = 3u;
constexpr std::array<std::uint32_t, 5> kBlockSizeCandidates{
    128u, 192u, 256u, 384u, 512u};

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

__device__ RT_FORCE_INLINE PackedPathState *
wavefront_shared_path_states(unsigned char *dynamic_shared,
                             std::uint32_t block_size) {
    constexpr std::size_t alignment = alignof(PackedPathState);
    const std::size_t queues =
        2u * static_cast<std::size_t>(block_size) * sizeof(std::uint32_t);
    const std::size_t offset =
        (queues + alignment - 1u) / alignment * alignment;
    return reinterpret_cast<PackedPathState *>(dynamic_shared + offset);
}

struct WavefrontBlockScratch {
    unsigned *queues;
    std::uint32_t *counts;
    unsigned long long *status_partials;
    unsigned long long *traversal_partial;
    unsigned long long *shadow_partial;
    unsigned long long *invalid_partial;
    unsigned long long *clamped_partial;
    unsigned long long *depth_steps_partial;
    unsigned long long *active_path_partial;
};

__device__ RT_FORCE_INLINE void process_wavefront_chunk(
    DeviceSceneView scene, PackedTransportSettings transport,
    std::uint32_t width, std::uint32_t height,
    std::uint32_t pixel_offset, std::uint32_t path_count,
    std::uint32_t local_base, std::uint32_t state_index_base,
    std::uint32_t sample_begin, std::uint32_t sample_end,
    std::uint32_t seed, PackedPathState *states, float sample_clamp,
    CudaFilmPixel *film, WavefrontBlockScratch scratch) {
    const unsigned thread = threadIdx.x;
    const std::uint32_t local_index = local_base + thread;
    const std::uint32_t state_slot = state_index_base + thread;
    const bool valid = local_index < path_count;
    const std::uint32_t pixel_index = pixel_offset + local_index;
    const std::uint32_t pixel_x = pixel_index % width;
    const std::uint32_t pixel_y = pixel_index / width;

    // One chunk advances a contiguous sample range for every path slot.
    // Per-sample block-wide barriers preserve the original wavefront order,
    // so film updates remain race-free and bit-identical to one launch per
    // sample.
    for (std::uint32_t sample = sample_begin; sample < sample_end;
         ++sample) {
        if (valid) {
            const std::uint32_t sample_seed =
                packed_transport::packed_camera_sample_seed(
                    seed, pixel_index, sample);
            RNG rng(sample_seed);
            const PackedRay ray =
                packed_transport::generate_packed_camera_ray_core(
                    scene.scene.camera, pixel_x, pixel_y, width, height,
                    rng);
            packed_transport::initialize_packed_path_state(
                scene.scene, ray, transport, rng.state, pixel_index,
                sample, states[state_slot]);
            if (states[state_slot].active()) {
                const std::uint32_t slot = atomicAdd(&scratch.counts[0], 1u);
                scratch.queues[slot] = state_slot;
            }
        }
        __syncthreads();

        std::uint32_t active = scratch.counts[0];
        std::uint32_t parity = 0;
        while (active != 0u) {
            const std::uint32_t next_parity = parity ^ 1u;
            const unsigned *source_queue =
                scratch.queues + parity * blockDim.x;
            unsigned *next_queue =
                scratch.queues + next_parity * blockDim.x;
            if (thread < active) {
                const std::uint32_t path_index = source_queue[thread];
                PackedPathState state = states[path_index];
                packed_transport::advance_packed_path_core(
                    scene.scene, transport, state);
                states[path_index] = state;
                if (state.active()) {
                    const std::uint32_t output_index =
                        atomicAdd(&scratch.counts[next_parity], 1u);
                    next_queue[output_index] = path_index;
                }
            }
            __syncthreads();
            active = scratch.counts[next_parity];
            __syncthreads();
            if (thread == 0) {
                *scratch.depth_steps_partial += 1u;
                *scratch.active_path_partial += active;
                scratch.counts[next_parity] = 0;
            }
            __syncthreads();
            parity = next_parity;
        }
        __syncthreads();

        if (valid) {
            const PackedPathState &state = states[state_slot];
            Float3 radiance = state.radiance;
            std::uint32_t status_index =
                static_cast<std::uint32_t>(state.status);
            if (status_index >= 8) {
                status_index = static_cast<std::uint32_t>(
                    PackedTransportStatus::InvalidInput);
            }

            atomicAdd(&scratch.status_partials[status_index], 1ull);
            atomicAdd(scratch.traversal_partial,
                      static_cast<unsigned long long>(state.traversal_steps));
            atomicAdd(scratch.shadow_partial,
                      static_cast<unsigned long long>(state.shadow_rays));

            if (state.status != PackedTransportStatus::Success ||
                !packed_transport::math::finite(radiance)) {
                radiance = {};
                atomicAdd(scratch.invalid_partial, 1ull);
            } else if (sample_clamp > 0.0f) {
                const float sample_luminance = luminance(radiance);
                if (sample_luminance > sample_clamp) {
                    const float scale = sample_clamp / sample_luminance;
                    radiance =
                        packed_transport::math::multiply(radiance, scale);
                    atomicAdd(scratch.clamped_partial, 1ull);
                }
            }

            CudaFilmPixel &pixel = film[state.pixel_index];
            pixel.radiance =
                packed_transport::math::add(pixel.radiance, radiance);
            ++pixel.sample_count;
        }
        __syncthreads();
        if (thread == 0) {
            scratch.counts[0] = 0;
            scratch.counts[1] = 0;
        }
        __syncthreads();
    }
}

__device__ RT_FORCE_INLINE WavefrontBlockScratch
make_wavefront_block_scratch(unsigned *queues, std::uint32_t *counts,
                             unsigned long long *status_partials,
                             unsigned long long *traversal_partial,
                             unsigned long long *shadow_partial,
                             unsigned long long *invalid_partial,
                             unsigned long long *clamped_partial,
                             unsigned long long *depth_steps_partial,
                             unsigned long long *active_path_partial) {
    return {queues,
            counts,
            status_partials,
            traversal_partial,
            shadow_partial,
            invalid_partial,
            clamped_partial,
            depth_steps_partial,
            active_path_partial};
}

__device__ RT_FORCE_INLINE void initialize_wavefront_block_scratch(
    WavefrontBlockScratch scratch) {
    const unsigned thread = threadIdx.x;
    if (thread < 8) {
        scratch.status_partials[thread] = 0;
    }
    if (thread == 0) {
        scratch.counts[0] = 0;
        scratch.counts[1] = 0;
        *scratch.traversal_partial = 0;
        *scratch.shadow_partial = 0;
        *scratch.invalid_partial = 0;
        *scratch.clamped_partial = 0;
        *scratch.depth_steps_partial = 0;
        *scratch.active_path_partial = 0;
    }
    __syncthreads();
}

__device__ RT_FORCE_INLINE void flush_wavefront_block_scratch(
    WavefrontBlockScratch scratch, DeviceRenderCounters *counters) {
    __syncthreads();
    if (threadIdx.x == 0) {
        for (unsigned index = 0; index < 8; ++index) {
            atomicAdd(&counters->status_counts[index],
                      scratch.status_partials[index]);
        }
        atomicAdd(&counters->traversal_steps, *scratch.traversal_partial);
        atomicAdd(&counters->shadow_rays, *scratch.shadow_partial);
        atomicAdd(&counters->invalid_samples, *scratch.invalid_partial);
        atomicAdd(&counters->clamped_samples, *scratch.clamped_partial);
        atomicAdd(&counters->depth_steps, *scratch.depth_steps_partial);
        atomicAdd(&counters->active_path_steps,
                  *scratch.active_path_partial);
    }
}

__global__ void render_paths_kernel(
    DeviceSceneView scene, PackedTransportSettings transport,
    std::uint32_t width, std::uint32_t height,
    std::uint32_t pixel_offset, std::uint32_t path_count,
    std::uint32_t sample_begin, std::uint32_t sample_end,
    std::uint32_t seed, PackedPathState *states, bool shared_path_state,
    float sample_clamp, CudaFilmPixel *film,
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

    const WavefrontBlockScratch scratch = make_wavefront_block_scratch(
        queues, counts, status_partials, &traversal_partial,
        &shadow_partial, &invalid_partial, &clamped_partial,
        &depth_steps_partial, &active_path_partial);
    initialize_wavefront_block_scratch(scratch);

    PackedPathState *path_states =
        shared_path_state
            ? wavefront_shared_path_states(dynamic_shared, blockDim.x)
            : states;
    process_wavefront_chunk(
        scene, transport, width, height, pixel_offset, path_count,
        blockIdx.x * blockDim.x,
        shared_path_state ? 0u : blockIdx.x * blockDim.x, sample_begin,
        sample_end, seed, path_states, sample_clamp, film, scratch);
    flush_wavefront_block_scratch(scratch, counters);
}

__global__ void render_paths_persistent_kernel(
    DeviceSceneView scene, PackedTransportSettings transport,
    std::uint32_t width, std::uint32_t height,
    std::uint32_t pixel_count, std::uint32_t chunk_size,
    std::uint32_t sample_begin, std::uint32_t sample_end,
    std::uint32_t seed, PackedPathState *states, bool shared_path_state,
    float sample_clamp, CudaFilmPixel *film,
    DeviceRenderCounters *counters, std::uint32_t *work_counter) {
    extern __shared__ unsigned char dynamic_shared[];
    unsigned *queues = reinterpret_cast<unsigned *>(dynamic_shared);

    __shared__ std::uint32_t counts[2];
    __shared__ std::uint32_t next_chunk;
    __shared__ unsigned long long status_partials[8];
    __shared__ unsigned long long traversal_partial;
    __shared__ unsigned long long shadow_partial;
    __shared__ unsigned long long invalid_partial;
    __shared__ unsigned long long clamped_partial;
    __shared__ unsigned long long depth_steps_partial;
    __shared__ unsigned long long active_path_partial;

    const WavefrontBlockScratch scratch = make_wavefront_block_scratch(
        queues, counts, status_partials, &traversal_partial,
        &shadow_partial, &invalid_partial, &clamped_partial,
        &depth_steps_partial, &active_path_partial);
    initialize_wavefront_block_scratch(scratch);

    PackedPathState *block_states =
        shared_path_state
            ? wavefront_shared_path_states(dynamic_shared, blockDim.x)
            : states + blockIdx.x * chunk_size;
    for (;;) {
        if (threadIdx.x == 0) {
            next_chunk = atomicAdd(work_counter, 1u);
        }
        __syncthreads();
        const std::uint32_t chunk = next_chunk;
        const std::uint32_t pixel_offset = chunk * chunk_size;
        if (pixel_offset >= pixel_count) {
            break;
        }
        const std::uint32_t count =
            min(pixel_count - pixel_offset, chunk_size);
        process_wavefront_chunk(
            scene, transport, width, height, pixel_offset, count, 0, 0,
            sample_begin, sample_end, seed, block_states, sample_clamp, film,
            scratch);
    }
    flush_wavefront_block_scratch(scratch, counters);
}


std::size_t wavefront_queue_bytes(std::uint32_t block_size) {
    return 2u * static_cast<std::size_t>(block_size) * sizeof(std::uint32_t);
}

std::size_t wavefront_shared_bytes(std::uint32_t block_size,
                                   bool shared_path_state) {
    if (!shared_path_state) {
        return wavefront_queue_bytes(block_size);
    }
    constexpr std::size_t alignment = alignof(PackedPathState);
    const std::size_t queues = wavefront_queue_bytes(block_size);
    const std::size_t offset = (queues + alignment - 1u) / alignment * alignment;
    return offset +
           static_cast<std::size_t>(block_size) * sizeof(PackedPathState);
}

int wavefront_active_blocks_per_sm(std::uint32_t block_size,
                                   bool shared_path_state) {
    int active_blocks = 0;
    const cudaError_t status =
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks, render_paths_kernel,
            static_cast<int>(block_size),
            wavefront_shared_bytes(block_size, shared_path_state));
    if (status != cudaSuccess || active_blocks == 0) {
        return -1;
    }
    return active_blocks;
}

bool wavefront_block_size_launchable(std::uint32_t block_size,
                                     bool shared_path_state) {
    return wavefront_active_blocks_per_sm(block_size, shared_path_state) > 0;
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
    if (settings.block_size > 1024) {
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

std::uint32_t choose_samples_per_launch(
    const CudaRenderSettings &settings) {
    const std::uint32_t requested =
        settings.samples_per_launch == 0 ? 8u : settings.samples_per_launch;
    return std::max(1u, std::min(requested, settings.samples_per_pixel));
}

std::uint32_t choose_batch_size(const CudaRenderSettings &settings,
                                std::uint32_t pixel_count) {
    const std::uint32_t requested = settings.batch_size == 0
                                        ? kDefaultBatchSize
                                        : settings.batch_size;
    return std::max(1u, std::min(requested, pixel_count));
}

std::uint32_t choose_work_chunk_size(const CudaRenderSettings &settings,
                                       std::uint32_t block_size) {
    // A chunk equal to the block size keeps every thread usefully
    // occupied. The 32/64 pixel variants from the plan are available
    // through --cuda-work-chunk-size, but the full block variant measured
    // fastest on the 4060.
    const std::uint32_t requested = settings.work_chunk_size == 0u
                                        ? block_size
                                        : settings.work_chunk_size;
    return std::max(1u, std::min(requested, block_size));
}

std::uint32_t choose_persistent_grid(std::uint32_t block_size,
                                     std::uint32_t pixel_count,
                                     std::uint32_t chunk_size,
                                     bool shared_path_state) {
    const int active_blocks =
        wavefront_active_blocks_per_sm(block_size, shared_path_state);
    if (active_blocks <= 0) {
        throw std::runtime_error(
            "CUDA persistent grid cannot determine blocks per SM");
    }
    int device = 0;
    RT_CUDA_CHECK(cudaGetDevice(&device));
    int multiprocessor_count = 0;
    RT_CUDA_CHECK(cudaDeviceGetAttribute(
        &multiprocessor_count, cudaDevAttrMultiProcessorCount, device));
    const std::uint32_t ideal =
        static_cast<std::uint32_t>(active_blocks) *
        static_cast<std::uint32_t>(multiprocessor_count);
    const std::uint32_t needed =
        (pixel_count + chunk_size - 1u) / chunk_size;
    return std::max(1u, std::min(ideal, needed));
}

std::uint32_t choose_block_size_heuristic(bool shared_path_state) {
    std::uint32_t best = kBlockSizeFallback;
    int best_occupancy = 0;
    for (const std::uint32_t candidate : kBlockSizeCandidates) {
        const int active_blocks = wavefront_active_blocks_per_sm(
            candidate, shared_path_state);
        if (active_blocks <= 0) {
            continue;
        }
        const int occupancy =
            active_blocks * static_cast<int>(candidate);
        // Maximize resident threads. On the current kernel all candidate
        // sizes that fit are register-limited; when occupancy ties, more
        // smaller blocks per SM hide latency and tail effects better than
        // fewer larger blocks, so ties keep the first (smallest) candidate.
        if (occupancy > best_occupancy) {
            best_occupancy = occupancy;
            best = candidate;
        }
    }
    return best;
}

struct WavefrontLaunchResult {
    float milliseconds = 0.0f;
    std::uint32_t batch_count = 0;
    bool cancelled = false;
};

WavefrontLaunchResult launch_wavefront_kernels(
    DeviceSceneView scene, const CudaRenderSettings &settings,
    const std::atomic<bool> *cancel, std::uint32_t block_size,
    std::uint32_t pixel_count, std::uint32_t batch_size,
    std::uint32_t samples_per_launch, bool persistent_grid,
    std::uint32_t chunk_size, std::uint32_t grid_size,
    CudaFilmPixel *film, PackedPathState *states,
    DeviceRenderCounters *counters, std::uint32_t *work_counter) {
    RT_CUDA_CHECK(cudaMemset(film, 0,
                             static_cast<std::size_t>(pixel_count) *
                                 sizeof(CudaFilmPixel)));
    RT_CUDA_CHECK(cudaMemset(counters, 0,
                             sizeof(DeviceRenderCounters)));

    CudaEvent begin;
    CudaEvent end;
    RT_CUDA_CHECK(cudaEventRecord(begin));

    WavefrontLaunchResult result;
    const std::size_t shared_bytes = wavefront_shared_bytes(
        block_size, settings.shared_path_state);
    if (persistent_grid) {
        const std::uint32_t chunks_per_sample_group =
            (pixel_count + chunk_size - 1u) / chunk_size;
        for (std::uint32_t sample = 0;
             sample < settings.samples_per_pixel && !result.cancelled;
             sample += samples_per_launch) {
            if (cancel != nullptr && cancel->load()) {
                result.cancelled = true;
                break;
            }
            const std::uint32_t sample_end =
                std::min(sample + samples_per_launch,
                         settings.samples_per_pixel);
            RT_CUDA_CHECK(cudaMemset(work_counter, 0,
                                     sizeof(std::uint32_t)));
            render_paths_persistent_kernel<<<grid_size, block_size,
                                              shared_bytes>>>(
                scene, settings.transport, settings.width, settings.height,
                pixel_count, chunk_size, sample, sample_end, settings.seed,
                states, settings.shared_path_state,
                static_cast<float>(settings.sample_clamp), film, counters,
                work_counter);
            RT_CUDA_CHECK(cudaGetLastError());
            result.batch_count += chunks_per_sample_group;
        }
    } else {
        for (std::uint32_t sample = 0;
             sample < settings.samples_per_pixel && !result.cancelled;
             sample += samples_per_launch) {
            const std::uint32_t sample_end =
                std::min(sample + samples_per_launch,
                         settings.samples_per_pixel);
            for (std::uint32_t offset = 0; offset < pixel_count;
                 offset += batch_size) {
                if (cancel != nullptr && cancel->load()) {
                    result.cancelled = true;
                    break;
                }
                ++result.batch_count;
                const std::uint32_t count =
                    std::min(batch_size, pixel_count - offset);
                const std::uint32_t grid =
                    (count + block_size - 1) / block_size;
                render_paths_kernel<<<grid, block_size, shared_bytes>>>(
                    scene, settings.transport, settings.width,
                    settings.height, offset, count, sample, sample_end,
                    settings.seed, states, settings.shared_path_state,
                    static_cast<float>(settings.sample_clamp), film,
                    counters);
                RT_CUDA_CHECK(cudaGetLastError());
                // One kernel owns a complete pixel batch and sample group.
                // The default stream orders all batches, so the final event
                // sync is the only required host synchronization point.
            }
        }
    }
    RT_CUDA_CHECK(cudaEventRecord(end));
    RT_CUDA_CHECK(cudaEventSynchronize(end));
    RT_CUDA_CHECK(cudaEventElapsedTime(&result.milliseconds, begin, end));
    return result;
}


std::uint32_t float_bits(float value) {
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

struct WavefrontBlockTuningKey {
    int device = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t samples_per_pixel = 0;
    std::uint32_t samples_per_launch = 0;
    std::uint32_t batch_size = 0;
    std::uint32_t max_depth = 0;
    std::uint32_t policy_kind = 0;
    std::uint32_t policy_flags = 0;
    std::uint32_t rr_start_depth = 0;
    std::uint32_t rr_min_survival_bits = 0;
    std::uint32_t sample_clamp_bits = 0;
    std::uintptr_t bvh_nodes = 0;
    std::uintptr_t aggregate_nodes = 0;
    std::uintptr_t triangles = 0;
    std::uintptr_t materials = 0;
};

WavefrontBlockTuningKey make_block_tuning_key(
    const CudaRenderSettings &settings, std::uint32_t batch_size,
    std::uint32_t samples_per_launch, DeviceSceneView scene) {
    WavefrontBlockTuningKey key;
    RT_CUDA_CHECK(cudaGetDevice(&key.device));
    key.width = settings.width;
    key.height = settings.height;
    key.samples_per_pixel = settings.samples_per_pixel;
    key.samples_per_launch = samples_per_launch;
    key.batch_size = batch_size;
    key.max_depth = settings.transport.max_depth;
    key.policy_kind =
        static_cast<std::uint32_t>(settings.transport.policy.kind);
    key.policy_flags = settings.transport.policy.flags;
    key.rr_start_depth = settings.transport.policy.rr_start_depth;
    key.rr_min_survival_bits =
        float_bits(settings.transport.policy.rr_min_survival);
    key.sample_clamp_bits = float_bits(settings.sample_clamp);
    key.bvh_nodes =
        reinterpret_cast<std::uintptr_t>(scene.scene.bvh_nodes.data);
    key.aggregate_nodes =
        reinterpret_cast<std::uintptr_t>(scene.scene.aggregates.data);
    key.triangles =
        reinterpret_cast<std::uintptr_t>(scene.scene.triangles.data);
    key.materials =
        reinterpret_cast<std::uintptr_t>(scene.scene.materials.data);
    return key;
}

bool same_tuning_key(const WavefrontBlockTuningKey &lhs,
                     const WavefrontBlockTuningKey &rhs) {
    return lhs.device == rhs.device && lhs.width == rhs.width &&
           lhs.height == rhs.height &&
           lhs.samples_per_pixel == rhs.samples_per_pixel &&
           lhs.samples_per_launch == rhs.samples_per_launch &&
           lhs.batch_size == rhs.batch_size &&
           lhs.max_depth == rhs.max_depth &&
           lhs.policy_kind == rhs.policy_kind &&
           lhs.policy_flags == rhs.policy_flags &&
           lhs.rr_start_depth == rhs.rr_start_depth &&
           lhs.rr_min_survival_bits == rhs.rr_min_survival_bits &&
           lhs.sample_clamp_bits == rhs.sample_clamp_bits &&
           lhs.bvh_nodes == rhs.bvh_nodes &&
           lhs.aggregate_nodes == rhs.aggregate_nodes &&
           lhs.triangles == rhs.triangles &&
           lhs.materials == rhs.materials;
}

std::size_t workspace_bytes(std::uint32_t pixel_count,
                            std::uint32_t batch_size);

std::uint32_t autotune_block_size(
    DeviceSceneView scene, const CudaRenderSettings &settings,
    const std::atomic<bool> *cancel) {
    // Keep the tuning image small, but wide enough that even the largest
    // candidate can fill the whole GPU grid. A pure 64x36 image would bias
    // the result towards the smallest block size.
    const std::uint32_t tune_width =
        std::min(settings.width, kAutotuneTargetWidth);
    const std::uint32_t tune_height = std::max(
        1u, std::min(settings.height,
                     static_cast<std::uint32_t>(
                         static_cast<std::uint64_t>(settings.height) *
                         tune_width / settings.width)));
    const std::uint32_t tune_pixel_count = tune_width * tune_height;
    const std::uint32_t tune_batch_size =
        choose_batch_size(settings, tune_pixel_count);

    CudaRenderSettings tune_settings = settings;
    tune_settings.width = tune_width;
    tune_settings.height = tune_height;
    tune_settings.samples_per_pixel = 1;
    tune_settings.samples_per_launch = 1;
    tune_settings.batch_size = tune_batch_size;
    tune_settings.block_size = kBlockSizeFallback;
    tune_settings.autotune_block_size = false;
    tune_settings.persistent_grid = false;

    const std::size_t required =
        workspace_bytes(tune_pixel_count, tune_batch_size);
    ensure_cuda_workspace_fits(required, 0,
                               "CUDA block-size autotune workspace");
    DeviceBuffer<CudaFilmPixel> tune_film;
    DeviceBuffer<PackedPathState> tune_states;
    DeviceBuffer<DeviceRenderCounters> tune_counters;
    tune_film.allocate(tune_pixel_count);
    tune_states.allocate(tune_batch_size);
    tune_counters.allocate(1);

    auto run_candidate = [&](std::uint32_t candidate) {
        return launch_wavefront_kernels(
            scene, tune_settings, cancel, candidate, tune_pixel_count,
            tune_batch_size, 1, false, 0u, 0u, tune_film.data(),
            tune_states.data(), tune_counters.data(), nullptr);
    };

    auto is_cancelled = [&]() {
        return cancel != nullptr && cancel->load();
    };

    // Warm every launchable candidate before timing so the first
    // candidate does not pay the GPU clock/frequency ramp cost.
    for (const std::uint32_t candidate : kBlockSizeCandidates) {
        if (!wavefront_block_size_launchable(
                candidate, tune_settings.shared_path_state)) {
            continue;
        }
        if (is_cancelled()) {
            return kBlockSizeFallback;
        }
        run_candidate(candidate);
    }

    std::uint32_t best_block_size = kBlockSizeFallback;
    float best_milliseconds = std::numeric_limits<float>::infinity();
    for (const std::uint32_t candidate : kBlockSizeCandidates) {
        if (!wavefront_block_size_launchable(
                candidate, tune_settings.shared_path_state)) {
            continue;
        }
        float candidate_milliseconds =
            std::numeric_limits<float>::infinity();
        for (std::uint32_t run = 0;
             run < kAutotuneMeasurementRuns; ++run) {
            if (is_cancelled()) {
                return best_milliseconds ==
                               std::numeric_limits<float>::infinity()
                           ? kBlockSizeFallback
                           : best_block_size;
            }
            const WavefrontLaunchResult measurement = run_candidate(candidate);
            if (measurement.cancelled) {
                return best_milliseconds ==
                               std::numeric_limits<float>::infinity()
                           ? kBlockSizeFallback
                           : best_block_size;
            }
            candidate_milliseconds =
                std::min(candidate_milliseconds,
                         measurement.milliseconds);
        }
        if (candidate_milliseconds < best_milliseconds) {
            best_milliseconds = candidate_milliseconds;
            best_block_size = candidate;
        }
    }
    return best_block_size;
}

std::size_t workspace_bytes(std::uint32_t pixel_count,
                            std::uint32_t batch_size) {
    return static_cast<std::size_t>(pixel_count) * sizeof(CudaFilmPixel) +
           static_cast<std::size_t>(batch_size) * sizeof(PackedPathState) +
           sizeof(DeviceRenderCounters) + sizeof(std::uint32_t);
}

} // namespace

struct CudaRenderWorkspace::Impl {
    DeviceBuffer<CudaFilmPixel> film;
    DeviceBuffer<PackedPathState> states;
    DeviceBuffer<DeviceRenderCounters> counters;
    DeviceBuffer<std::uint32_t> work_counter;
    std::uint64_t generation = 0;

    struct BlockSizeTuning {
        WavefrontBlockTuningKey key;
        std::uint32_t block_size = 0;
    };
    BlockSizeTuning block_size_tuning;

    std::size_t allocated_bytes() const noexcept {
        return film.bytes() + states.bytes() + counters.bytes() +
               work_counter.bytes();
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
        changed |= work_counter.ensure_capacity_discard(1);
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

void CudaRenderWorkspace::invalidate_block_size_tuning() noexcept {
    if (m_impl) {
        m_impl->block_size_tuning = {};
    }
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
    const std::uint32_t samples_per_launch =
        choose_samples_per_launch(settings);
    CudaRenderWorkspace::Impl &buffers = *workspace.m_impl;

    std::uint32_t block_size = settings.block_size;
    if (block_size == 0) {
        if (settings.autotune_block_size) {
            const WavefrontBlockTuningKey key = make_block_tuning_key(
                settings, batch_size, samples_per_launch, scene);
            if (buffers.block_size_tuning.block_size != 0 &&
                same_tuning_key(buffers.block_size_tuning.key, key)) {
                block_size = buffers.block_size_tuning.block_size;
            } else {
                block_size = autotune_block_size(scene, settings, cancel);
                if (!wavefront_block_size_launchable(
                        block_size, settings.shared_path_state)) {
                    throw std::runtime_error(
                        "CUDA block-size autotune found no launchable "
                        "wavefront block size");
                }
                buffers.block_size_tuning = {key, block_size};
            }
        } else {
            block_size =
                choose_block_size_heuristic(settings.shared_path_state);
        }
    }
    if (!wavefront_block_size_launchable(block_size,
                                         settings.shared_path_state)) {
        throw std::invalid_argument(
            "CUDA render block size " + std::to_string(block_size) +
            " cannot launch with the current wavefront kernel");
    }

    const bool persistent_grid = settings.persistent_grid;
    const std::uint32_t chunk_size =
        persistent_grid ? choose_work_chunk_size(settings, block_size) : 0u;
    const std::uint32_t grid_size =
        persistent_grid
            ? choose_persistent_grid(block_size, pixel_count, chunk_size,
                                     settings.shared_path_state)
            : 0u;
    const std::uint32_t path_capacity =
        settings.shared_path_state
            ? 0u
            : (persistent_grid ? grid_size * chunk_size : batch_size);

    workspace.m_impl->ensure_capacity(pixel_count, path_capacity);

    const WavefrontLaunchResult launch = launch_wavefront_kernels(
        scene, settings, cancel, block_size, pixel_count, batch_size,
        samples_per_launch, persistent_grid, chunk_size, grid_size,
        buffers.film.data(), buffers.states.data(), buffers.counters.data(),
        buffers.work_counter.data());

    CudaRenderOutput output;
    buffers.film.download_prefix(output.film, pixel_count);
    std::vector<DeviceRenderCounters> host_counters;
    buffers.counters.download_prefix(host_counters, 1);
    const DeviceRenderCounters &counter = host_counters.front();
    output.stats.milliseconds = launch.milliseconds;
    output.stats.traversal_steps = counter.traversal_steps;
    output.stats.shadow_rays = counter.shadow_rays;
    output.stats.clamped_samples = counter.clamped_samples;
    output.stats.invalid_samples = counter.invalid_samples;
    output.stats.batch_size = batch_size;
    output.stats.batch_count = launch.batch_count;
    output.stats.samples_per_launch = samples_per_launch;
    output.stats.block_size = block_size;
    output.stats.work_chunk_size = chunk_size;
    output.stats.persistent_blocks = grid_size;
    output.stats.persistent_grid = persistent_grid;
    output.stats.advance_launches = counter.depth_steps;
    output.stats.active_path_steps = counter.active_path_steps;
    const CudaWorkspaceInfo workspace_info = workspace.info();
    output.stats.workspace_bytes = workspace_info.bytes;
    output.stats.workspace_generation = workspace_info.generation;
    output.stats.workspace_pixel_capacity = workspace_info.pixel_capacity;
    output.stats.workspace_path_capacity = workspace_info.path_capacity;
    output.stats.cancelled = launch.cancelled;
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

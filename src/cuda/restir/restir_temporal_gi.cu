#include "restir_temporal_gi.h"

#include "cuda_error.h"
#include "restir_gi_temporal_core.h"

namespace cuda_backend {
namespace {

__global__ void temporal_gi_kernel(
    DeviceSceneView scene, restir::RestirSurface *current_surfaces,
    const restir::RestirGIReservoir *current_reservoirs,
    const restir::RestirSurface *previous_surfaces,
    const restir::RestirGIReservoir *previous_reservoirs,
    PackedCamera previous_camera, bool history_available,
    restir::RestirGIReservoir *destination,
    std::uint32_t width, std::uint32_t height,
    std::uint32_t iteration, std::uint32_t seed,
    std::uint32_t max_history_length,
    std::uint32_t max_candidates, float normal_threshold,
    float depth_threshold, bool pairwise,
    PackedTransportSettings transport,
    DeviceRestirCounters *counters,
    std::uint32_t *status_output,
    CudaRestirStatsLevel stats_level) {
    __shared__ unsigned long long status_partial[16];
    __shared__ unsigned long long rejection_partial[16];
    __shared__ unsigned long long shift_partial[16];
    __shared__ unsigned long long candidates_partial;
    __shared__ unsigned long long accepted_partial;
    __shared__ unsigned long long pairwise_partial;
    __shared__ unsigned long long replay_evaluations_partial;
    __shared__ unsigned long long replay_shadow_partial;
    __shared__ unsigned long long replay_traversal_partial;

    const bool collect_any = restir_collects_any_stats(stats_level);
    const bool collect_full = restir_collects_full_stats(stats_level);
    const unsigned thread = threadIdx.x;
    if (thread < 16u) {
        status_partial[thread] = 0;
        rejection_partial[thread] = 0;
        shift_partial[thread] = 0;
    }
    if (thread == 0) {
        candidates_partial = 0;
        accepted_partial = 0;
        pairwise_partial = 0;
        replay_evaluations_partial = 0;
        replay_shadow_partial = 0;
        replay_traversal_partial = 0;
    }
    __syncthreads();

    const std::uint32_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel < width * height) {
        restir::RestirGITemporalStats stats;
        const restir::RestirGIStatus status = pairwise
            ? restir::temporal_resample_gi_pairwise(
                scene.scene, current_surfaces[pixel],
                current_reservoirs[pixel],
                history_available ? &previous_camera : nullptr,
                history_available ? previous_surfaces : nullptr,
                history_available ? previous_reservoirs : nullptr,
                width, height, pixel, iteration, seed, max_history_length,
                max_candidates, normal_threshold, depth_threshold,
                transport, destination[pixel], stats)
            : restir::temporal_resample_gi_basic(
                scene.scene, current_surfaces[pixel],
                current_reservoirs[pixel],
                history_available ? &previous_camera : nullptr,
                history_available ? previous_surfaces : nullptr,
                history_available ? previous_reservoirs : nullptr,
                width, height, pixel, iteration, seed, max_history_length,
                max_candidates, normal_threshold, depth_threshold,
                transport, destination[pixel], stats);
        std::uint32_t status_index = static_cast<std::uint32_t>(status);
        status_index = status_index < 16u ? status_index : 15u;
        if (status_output != nullptr) {
            status_output[pixel] = status_index;
        }
        if (collect_full) {
            atomicAdd(&status_partial[status_index], 1ull);
            const std::uint32_t rejection_index =
                static_cast<std::uint32_t>(stats.rejection);
            if (rejection_index < 16u) {
                atomicAdd(&rejection_partial[rejection_index], 1ull);
            }
            const std::uint32_t shift_index =
                static_cast<std::uint32_t>(stats.shift_failure);
            if (shift_index < 16u &&
                stats.shift_failure != restir::RestirGIShiftFailure::None) {
                atomicAdd(&shift_partial[shift_index], 1ull);
            }
        }
        if (collect_any) {
            atomicAdd(&candidates_partial,
                      static_cast<unsigned long long>(stats.candidates));
            atomicAdd(&accepted_partial,
                      static_cast<unsigned long long>(stats.accepted));
            atomicAdd(&pairwise_partial,
                      static_cast<unsigned long long>(
                          stats.pairwise_fallbacks));
            atomicAdd(&replay_evaluations_partial,
                      static_cast<unsigned long long>(
                          stats.replay_evaluations));
            atomicAdd(&replay_shadow_partial,
                      static_cast<unsigned long long>(
                          stats.replay_shadow_rays));
            atomicAdd(&replay_traversal_partial,
                      static_cast<unsigned long long>(
                          stats.replay_traversal_steps));
        }
    }

    __syncthreads();
    if (thread == 0) {
        if (collect_full) {
            for (unsigned index = 0; index < 16u; ++index) {
                atomicAdd(&counters->gi_temporal_status[index],
                          status_partial[index]);
                atomicAdd(&counters->gi_temporal_rejection[index],
                          rejection_partial[index]);
                atomicAdd(&counters->gi_shift_failures[index],
                          shift_partial[index]);
            }
        }
        if (collect_any) {
            atomicAdd(&counters->gi_temporal_candidates,
                      candidates_partial);
            atomicAdd(&counters->gi_temporal_accepted, accepted_partial);
            atomicAdd(&counters->gi_temporal_pairwise_fallbacks,
                      pairwise_partial);
            atomicAdd(&counters->gi_replay_evaluations,
                      replay_evaluations_partial);
            atomicAdd(&counters->gi_replay_shadow_rays,
                      replay_shadow_partial);
            atomicAdd(&counters->gi_replay_traversal_steps,
                      replay_traversal_partial);
        }
    }
}

} // namespace

void launch_restir_temporal_gi(
    DeviceSceneView scene, restir::RestirSurface *current_surfaces,
    const restir::RestirGIReservoir *current_reservoirs,
    const restir::RestirSurface *previous_surfaces,
    const restir::RestirGIReservoir *previous_reservoirs,
    PackedCamera previous_camera, bool history_available,
    restir::RestirGIReservoir *destination,
    std::uint32_t width, std::uint32_t height,
    std::uint32_t iteration, std::uint32_t seed,
    std::uint32_t max_history_length,
    std::uint32_t max_candidates, float normal_threshold,
    float depth_threshold, bool pairwise,
    const PackedTransportSettings &transport,
    DeviceRestirCounters *counters,
    std::uint32_t block_size, std::uint32_t *status_output,
    CudaRestirStatsLevel stats_level) {
    const std::uint32_t pixel_count = width * height;
    const std::uint32_t grid =
        (pixel_count + block_size - 1u) / block_size;
    temporal_gi_kernel<<<grid, block_size>>>(
        scene, current_surfaces, current_reservoirs, previous_surfaces,
        previous_reservoirs, previous_camera, history_available,
        destination, width, height, iteration, seed, max_history_length,
        max_candidates, normal_threshold, depth_threshold, pairwise,
        transport, counters, status_output, stats_level);
    RT_CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda_backend

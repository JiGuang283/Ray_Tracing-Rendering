#include "restir_spatial_gi.h"

#include "cuda_error.h"
#include "restir_gi_spatial_pairwise_core.h"

namespace cuda_backend {
namespace {

__global__ void spatial_gi_kernel(
    DeviceSceneView scene, const restir::RestirSurface *surfaces,
    const restir::RestirGIReservoir *source,
    restir::RestirGIReservoir *destination,
    std::uint32_t width, std::uint32_t height,
    std::uint32_t iteration, std::uint32_t pass_index,
    std::uint32_t seed, std::uint32_t neighbor_count,
    std::uint32_t max_candidates, float normal_threshold,
    float depth_threshold, bool pairwise,
    PackedTransportSettings transport,
    DeviceRestirCounters *counters,
    std::uint32_t *status_output,
    CudaRestirStatsLevel stats_level) {
    __shared__ unsigned long long status_partial[16];
    __shared__ unsigned long long compatibility_partial[9];
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
        shift_partial[thread] = 0;
    }
    if (thread < 9u) {
        compatibility_partial[thread] = 0;
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
        restir::RestirGISpatialStats stats;
        const restir::RestirGIStatus status = pairwise
            ? restir::spatial_resample_gi_pairwise(
                scene.scene, surfaces, source, width, height, pixel,
                iteration, pass_index, seed, neighbor_count,
                max_candidates, normal_threshold, depth_threshold,
                transport, destination[pixel], stats)
            : restir::spatial_resample_gi_basic(
                scene.scene, surfaces, source, width, height, pixel,
                iteration, pass_index, seed, neighbor_count,
                max_candidates, normal_threshold, depth_threshold,
                transport, destination[pixel], stats);
        std::uint32_t status_index = static_cast<std::uint32_t>(status);
        status_index = status_index < 16u ? status_index : 15u;
        if (status_output != nullptr) {
            status_output[pixel] = status_index;
        }
        if (collect_full) {
            atomicAdd(&status_partial[status_index], 1ull);
            for (std::uint32_t index = 0u;
                 index < static_cast<std::uint32_t>(
                             restir::RestirSpatialCompatibility::Count);
                 ++index) {
                atomicAdd(&compatibility_partial[index],
                          static_cast<unsigned long long>(
                              stats.compatibility[index]));
            }
            for (std::uint32_t index = 0u;
                 index < kRestirShiftFailureBuckets; ++index) {
                atomicAdd(&shift_partial[index],
                          static_cast<unsigned long long>(
                              stats.shift_failures[index]));
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
                atomicAdd(&counters->gi_spatial_status[index],
                          status_partial[index]);
                atomicAdd(&counters->gi_shift_failures[index],
                          shift_partial[index]);
            }
            for (unsigned index = 0; index < 9u; ++index) {
                atomicAdd(&counters->gi_spatial_compatibility[index],
                          compatibility_partial[index]);
            }
        }
        if (collect_any) {
            atomicAdd(&counters->gi_spatial_candidates,
                      candidates_partial);
            atomicAdd(&counters->gi_spatial_accepted, accepted_partial);
            atomicAdd(&counters->gi_spatial_pairwise_fallbacks,
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

void launch_restir_spatial_gi(
    DeviceSceneView scene, const restir::RestirSurface *surfaces,
    const restir::RestirGIReservoir *source,
    restir::RestirGIReservoir *destination,
    std::uint32_t width, std::uint32_t height,
    std::uint32_t iteration, std::uint32_t pass_index,
    std::uint32_t seed, std::uint32_t neighbor_count,
    std::uint32_t max_candidates, float normal_threshold,
    float depth_threshold, bool pairwise,
    const PackedTransportSettings &transport,
    DeviceRestirCounters *counters,
    std::uint32_t block_size, std::uint32_t *status_output,
    CudaRestirStatsLevel stats_level) {
    const std::uint32_t pixel_count = width * height;
    const std::uint32_t grid =
        (pixel_count + block_size - 1u) / block_size;
    spatial_gi_kernel<<<grid, block_size>>>(
        scene, surfaces, source, destination, width, height, iteration,
        pass_index, seed, neighbor_count, max_candidates,
        normal_threshold, depth_threshold, pairwise, transport,
        counters, status_output, stats_level);
    RT_CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda_backend

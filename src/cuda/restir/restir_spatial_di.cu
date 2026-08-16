#include "restir_spatial_di.h"

#include "cuda_error.h"
#include "restir_spatial_core.h"
#include "restir_spatial_pairwise_core.h"

namespace cuda_backend {
namespace {

__device__ std::uint32_t spatial_status_index(
    restir::RestirDIStatus status) {
    const std::uint32_t index = static_cast<std::uint32_t>(status);
    return index < 11u
               ? index
               : static_cast<std::uint32_t>(
                     restir::RestirDIStatus::NonFinite);
}

__global__ void spatial_di_basic_kernel(
    DeviceSceneView scene, const restir::RestirSurface *surfaces,
    const restir::RestirDIReservoir *source,
    restir::RestirDIReservoir *destination, std::uint32_t width,
    std::uint32_t height, std::uint32_t iteration,
    std::uint32_t pass_index, std::uint32_t seed,
    std::uint32_t neighbor_count, std::uint32_t max_candidates,
    float normal_threshold, float depth_threshold,
    DeviceRestirCounters *counters) {
    __shared__ unsigned long long status_partial[11];
    __shared__ unsigned long long compatibility_partial[9];
    __shared__ unsigned long long candidates_partial;
    __shared__ unsigned long long accepted_partial;
    __shared__ unsigned long long rejected_partial;

    const unsigned thread = threadIdx.x;
    if (thread < 11u) {
        status_partial[thread] = 0;
    }
    if (thread < 9u) {
        compatibility_partial[thread] = 0;
    }
    if (thread == 0) {
        candidates_partial = 0;
        accepted_partial = 0;
        rejected_partial = 0;
    }
    __syncthreads();

    const std::uint32_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel < width * height) {
        restir::RestirDISpatialStats stats;
        const restir::RestirDIStatus status =
            restir::spatial_resample_di_basic(
                scene.scene, surfaces, source, width, height, pixel,
                iteration, pass_index, seed, neighbor_count,
                max_candidates, normal_threshold, depth_threshold,
                destination[pixel], stats);
        atomicAdd(&status_partial[spatial_status_index(status)], 1ull);
        atomicAdd(&candidates_partial,
                  static_cast<unsigned long long>(stats.candidates));
        atomicAdd(&accepted_partial,
                  static_cast<unsigned long long>(stats.accepted));
        atomicAdd(&rejected_partial,
                  static_cast<unsigned long long>(stats.rejected));
        for (std::uint32_t index = 0;
             index < static_cast<std::uint32_t>(
                         restir::RestirSpatialCompatibility::Count);
             ++index) {
            if (stats.compatibility[index] != 0u) {
                atomicAdd(&compatibility_partial[index],
                          static_cast<unsigned long long>(
                              stats.compatibility[index]));
            }
        }
    }

    __syncthreads();
    if (thread == 0) {
        for (unsigned index = 0; index < 11u; ++index) {
            atomicAdd(&counters->di_spatial_status[index],
                      status_partial[index]);
        }
        for (unsigned index = 0; index < 9u; ++index) {
            atomicAdd(&counters->spatial_compatibility[index],
                      compatibility_partial[index]);
        }
        atomicAdd(&counters->spatial_candidates, candidates_partial);
        atomicAdd(&counters->spatial_accepted, accepted_partial);
        atomicAdd(&counters->spatial_rejected, rejected_partial);
    }
}

__global__ void spatial_di_pairwise_kernel(
    DeviceSceneView scene, const restir::RestirSurface *surfaces,
    const restir::RestirDIReservoir *source,
    restir::RestirDIReservoir *destination, std::uint32_t width,
    std::uint32_t height, std::uint32_t iteration,
    std::uint32_t pass_index, std::uint32_t seed,
    std::uint32_t neighbor_count, std::uint32_t max_candidates,
    float normal_threshold, float depth_threshold,
    DeviceRestirCounters *counters) {
    __shared__ unsigned long long status_partial[11];
    __shared__ unsigned long long compatibility_partial[9];
    __shared__ unsigned long long candidates_partial;
    __shared__ unsigned long long accepted_partial;
    __shared__ unsigned long long rejected_partial;
    __shared__ unsigned long long pairwise_partial;

    const unsigned thread = threadIdx.x;
    if (thread < 11u) {
        status_partial[thread] = 0;
    }
    if (thread < 9u) {
        compatibility_partial[thread] = 0;
    }
    if (thread == 0) {
        candidates_partial = 0;
        accepted_partial = 0;
        rejected_partial = 0;
        pairwise_partial = 0;
    }
    __syncthreads();

    const std::uint32_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel < width * height) {
        restir::RestirDISpatialStats stats;
        const restir::RestirDIStatus status =
            restir::spatial_resample_di_pairwise(
                scene.scene, surfaces, source, width, height, pixel,
                iteration, pass_index, seed, neighbor_count,
                max_candidates, normal_threshold, depth_threshold,
                destination[pixel], stats);
        atomicAdd(&status_partial[spatial_status_index(status)], 1ull);
        atomicAdd(&candidates_partial,
                  static_cast<unsigned long long>(stats.candidates));
        atomicAdd(&accepted_partial,
                  static_cast<unsigned long long>(stats.accepted));
        atomicAdd(&rejected_partial,
                  static_cast<unsigned long long>(stats.rejected));
        atomicAdd(&pairwise_partial,
                  static_cast<unsigned long long>(stats.pairwise_fallbacks));
        for (std::uint32_t index = 0;
             index < static_cast<std::uint32_t>(
                         restir::RestirSpatialCompatibility::Count);
             ++index) {
            if (stats.compatibility[index] != 0u) {
                atomicAdd(&compatibility_partial[index],
                          static_cast<unsigned long long>(
                              stats.compatibility[index]));
            }
        }
    }

    __syncthreads();
    if (thread == 0) {
        for (unsigned index = 0; index < 11u; ++index) {
            atomicAdd(&counters->di_spatial_status[index],
                      status_partial[index]);
        }
        for (unsigned index = 0; index < 9u; ++index) {
            atomicAdd(&counters->spatial_compatibility[index],
                      compatibility_partial[index]);
        }
        atomicAdd(&counters->spatial_candidates, candidates_partial);
        atomicAdd(&counters->spatial_accepted, accepted_partial);
        atomicAdd(&counters->spatial_rejected, rejected_partial);
        atomicAdd(&counters->pairwise_fallbacks, pairwise_partial);
    }
}

} // namespace

void launch_restir_spatial_di_basic(
    DeviceSceneView scene, const restir::RestirSurface *surfaces,
    const restir::RestirDIReservoir *source,
    restir::RestirDIReservoir *destination, std::uint32_t width,
    std::uint32_t height, std::uint32_t iteration,
    std::uint32_t pass_index, std::uint32_t seed,
    std::uint32_t neighbor_count, std::uint32_t max_candidates,
    float normal_threshold, float depth_threshold,
    DeviceRestirCounters *counters, std::uint32_t block_size) {
    const std::uint32_t pixel_count = width * height;
    const std::uint32_t grid =
        (pixel_count + block_size - 1u) / block_size;
    spatial_di_basic_kernel<<<grid, block_size>>>(
        scene, surfaces, source, destination, width, height, iteration,
        pass_index, seed, neighbor_count, max_candidates, normal_threshold,
        depth_threshold, counters);
    RT_CUDA_CHECK(cudaGetLastError());
}

void launch_restir_spatial_di_pairwise(
    DeviceSceneView scene, const restir::RestirSurface *surfaces,
    const restir::RestirDIReservoir *source,
    restir::RestirDIReservoir *destination, std::uint32_t width,
    std::uint32_t height, std::uint32_t iteration,
    std::uint32_t pass_index, std::uint32_t seed,
    std::uint32_t neighbor_count, std::uint32_t max_candidates,
    float normal_threshold, float depth_threshold,
    DeviceRestirCounters *counters, std::uint32_t block_size) {
    const std::uint32_t pixel_count = width * height;
    const std::uint32_t grid =
        (pixel_count + block_size - 1u) / block_size;
    spatial_di_pairwise_kernel<<<grid, block_size>>>(
        scene, surfaces, source, destination, width, height, iteration,
        pass_index, seed, neighbor_count, max_candidates, normal_threshold,
        depth_threshold, counters);
    RT_CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda_backend

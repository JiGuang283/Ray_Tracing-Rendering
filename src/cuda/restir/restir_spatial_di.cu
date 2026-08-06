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
    const std::uint32_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= width * height) {
        return;
    }
    restir::RestirDISpatialStats stats;
    const restir::RestirDIStatus status =
        restir::spatial_resample_di_basic(
            scene.scene, surfaces, source, width, height, pixel, iteration,
            pass_index, seed, neighbor_count, max_candidates,
            normal_threshold, depth_threshold, destination[pixel], stats);
    atomicAdd(&counters->di_spatial_status[spatial_status_index(status)],
              1ull);
    atomicAdd(&counters->spatial_candidates,
              static_cast<unsigned long long>(stats.candidates));
    atomicAdd(&counters->spatial_accepted,
              static_cast<unsigned long long>(stats.accepted));
    atomicAdd(&counters->spatial_rejected,
              static_cast<unsigned long long>(stats.rejected));
    for (std::uint32_t index = 0;
         index < static_cast<std::uint32_t>(
                     restir::RestirSpatialCompatibility::Count);
         ++index) {
        if (stats.compatibility[index] != 0u) {
            atomicAdd(&counters->spatial_compatibility[index],
                      static_cast<unsigned long long>(
                          stats.compatibility[index]));
        }
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
    const std::uint32_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= width * height) {
        return;
    }
    restir::RestirDISpatialStats stats;
    const restir::RestirDIStatus status =
        restir::spatial_resample_di_pairwise(
            scene.scene, surfaces, source, width, height, pixel, iteration,
            pass_index, seed, neighbor_count, max_candidates,
            normal_threshold, depth_threshold, destination[pixel], stats);
    atomicAdd(&counters->di_spatial_status[spatial_status_index(status)],
              1ull);
    atomicAdd(&counters->spatial_candidates,
              static_cast<unsigned long long>(stats.candidates));
    atomicAdd(&counters->spatial_accepted,
              static_cast<unsigned long long>(stats.accepted));
    atomicAdd(&counters->spatial_rejected,
              static_cast<unsigned long long>(stats.rejected));
    atomicAdd(&counters->pairwise_fallbacks,
              static_cast<unsigned long long>(stats.pairwise_fallbacks));
    for (std::uint32_t index = 0;
         index < static_cast<std::uint32_t>(
                     restir::RestirSpatialCompatibility::Count);
         ++index) {
        if (stats.compatibility[index] != 0u) {
            atomicAdd(&counters->spatial_compatibility[index],
                      static_cast<unsigned long long>(
                          stats.compatibility[index]));
        }
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

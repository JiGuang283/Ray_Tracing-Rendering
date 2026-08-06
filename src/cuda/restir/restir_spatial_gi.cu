#include "restir_spatial_gi.h"

#include "cuda_error.h"
#include "restir_gi_spatial_core.h"

namespace cuda_backend {
namespace {

__global__ void spatial_gi_basic_kernel(
    DeviceSceneView scene, const restir::RestirSurface *surfaces,
    const restir::RestirGIReservoir *source,
    restir::RestirGIReservoir *destination,
    std::uint32_t width, std::uint32_t height,
    std::uint32_t iteration, std::uint32_t pass_index,
    std::uint32_t seed, std::uint32_t neighbor_count,
    std::uint32_t max_candidates, float normal_threshold,
    float depth_threshold, DeviceRestirCounters *counters,
    std::uint32_t *status_output) {
    const std::uint32_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= width * height) {
        return;
    }
    restir::RestirGISpatialStats stats;
    const restir::RestirGIStatus status =
        restir::spatial_resample_gi_basic(
            scene.scene, surfaces, source, width, height, pixel, iteration,
            pass_index, seed, neighbor_count, max_candidates,
            normal_threshold, depth_threshold, destination[pixel], stats);
    std::uint32_t status_index = static_cast<std::uint32_t>(status);
    status_index = status_index < 16u ? status_index : 15u;
    if (status_output != nullptr) {
        status_output[pixel] = status_index;
    }
    atomicAdd(&counters->gi_spatial_status[status_index], 1ull);
    atomicAdd(&counters->gi_spatial_candidates,
              static_cast<unsigned long long>(stats.candidates));
    atomicAdd(&counters->gi_spatial_accepted,
              static_cast<unsigned long long>(stats.accepted));
    for (std::uint32_t index = 0u;
         index < static_cast<std::uint32_t>(
                     restir::RestirSpatialCompatibility::Count);
         ++index) {
        atomicAdd(&counters->gi_spatial_compatibility[index],
                  static_cast<unsigned long long>(stats.compatibility[index]));
    }
    for (std::uint32_t index = 0u;
         index < kRestirShiftFailureBuckets; ++index) {
        atomicAdd(&counters->gi_shift_failures[index],
                  static_cast<unsigned long long>(stats.shift_failures[index]));
    }
}

} // namespace

void launch_restir_spatial_gi_basic(
    DeviceSceneView scene, const restir::RestirSurface *surfaces,
    const restir::RestirGIReservoir *source,
    restir::RestirGIReservoir *destination,
    std::uint32_t width, std::uint32_t height,
    std::uint32_t iteration, std::uint32_t pass_index,
    std::uint32_t seed, std::uint32_t neighbor_count,
    std::uint32_t max_candidates, float normal_threshold,
    float depth_threshold, DeviceRestirCounters *counters,
    std::uint32_t block_size, std::uint32_t *status_output) {
    const std::uint32_t pixel_count = width * height;
    const std::uint32_t grid =
        (pixel_count + block_size - 1u) / block_size;
    spatial_gi_basic_kernel<<<grid, block_size>>>(
        scene, surfaces, source, destination, width, height, iteration,
        pass_index, seed, neighbor_count, max_candidates,
        normal_threshold, depth_threshold, counters, status_output);
    RT_CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda_backend

#include "restir_temporal_di.h"

#include "cuda_error.h"
#include "restir_temporal_core.h"

namespace cuda_backend {
namespace {

__device__ std::uint32_t temporal_status_index(
    restir::RestirDIStatus status) {
    const std::uint32_t index = static_cast<std::uint32_t>(status);
    return index < 11u
               ? index
               : static_cast<std::uint32_t>(
                     restir::RestirDIStatus::NonFinite);
}

__global__ void temporal_di_kernel(
    DeviceSceneView scene, restir::RestirSurface *current_surfaces,
    const restir::RestirDIReservoir *current_reservoirs,
    const restir::RestirSurface *previous_surfaces,
    const restir::RestirDIReservoir *previous_reservoirs,
    PackedCamera previous_camera, bool history_available,
    restir::RestirDIReservoir *output, std::uint32_t width,
    std::uint32_t height, std::uint32_t iteration, std::uint32_t seed,
    std::uint32_t max_history_length, std::uint32_t max_candidates,
    float normal_threshold, float depth_threshold, bool pairwise,
    DeviceRestirCounters *counters) {
    const std::uint32_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= width * height) {
        return;
    }
    restir::RestirDITemporalStats stats;
    const PackedCamera *history_camera =
        history_available ? &previous_camera : nullptr;
    const restir::RestirSurface *history_surfaces =
        history_available ? previous_surfaces : nullptr;
    const restir::RestirDIReservoir *history_reservoirs =
        history_available ? previous_reservoirs : nullptr;
    const restir::RestirDIStatus status =
        pairwise
            ? restir::temporal_resample_di_pairwise(
                  scene.scene, current_surfaces[pixel],
                  current_reservoirs[pixel], history_camera,
                  history_surfaces, history_reservoirs, width, height, pixel,
                  iteration, seed, max_history_length, max_candidates,
                  normal_threshold, depth_threshold, output[pixel], stats)
            : restir::temporal_resample_di_basic(
                  scene.scene, current_surfaces[pixel],
                  current_reservoirs[pixel], history_camera,
                  history_surfaces, history_reservoirs, width, height, pixel,
                  iteration, seed, max_history_length, max_candidates,
                  normal_threshold, depth_threshold, output[pixel], stats);
    atomicAdd(&counters->di_temporal_status[temporal_status_index(status)],
              1ull);
    atomicAdd(&counters->temporal_rejection[
                  static_cast<std::uint32_t>(stats.rejection)],
              1ull);
    atomicAdd(&counters->temporal_candidates,
              static_cast<unsigned long long>(stats.candidates));
    atomicAdd(&counters->temporal_accepted,
              static_cast<unsigned long long>(stats.accepted));
    atomicAdd(&counters->temporal_pairwise_fallbacks,
              static_cast<unsigned long long>(stats.pairwise_fallbacks));
}

} // namespace

void launch_restir_temporal_di(
    DeviceSceneView scene, restir::RestirSurface *current_surfaces,
    const restir::RestirDIReservoir *current_reservoirs,
    const restir::RestirSurface *previous_surfaces,
    const restir::RestirDIReservoir *previous_reservoirs,
    PackedCamera previous_camera, bool history_available,
    restir::RestirDIReservoir *output, std::uint32_t width,
    std::uint32_t height, std::uint32_t iteration, std::uint32_t seed,
    std::uint32_t max_history_length, std::uint32_t max_candidates,
    float normal_threshold, float depth_threshold, bool pairwise,
    DeviceRestirCounters *counters, std::uint32_t block_size) {
    const std::uint32_t pixel_count = width * height;
    const std::uint32_t grid =
        (pixel_count + block_size - 1u) / block_size;
    temporal_di_kernel<<<grid, block_size>>>(
        scene, current_surfaces, current_reservoirs, previous_surfaces,
        previous_reservoirs, previous_camera, history_available, output,
        width, height, iteration, seed, max_history_length, max_candidates,
        normal_threshold, depth_threshold, pairwise, counters);
    RT_CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda_backend

#include "restir_temporal_gi.h"

#include "cuda_error.h"
#include "restir_gi_temporal_core.h"

namespace cuda_backend {
namespace {

__global__ void temporal_gi_basic_kernel(
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
    float depth_threshold, DeviceRestirCounters *counters,
    std::uint32_t *status_output) {
    const std::uint32_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= width * height) {
        return;
    }
    restir::RestirGITemporalStats stats;
    const restir::RestirGIStatus status =
        restir::temporal_resample_gi_basic(
            scene.scene, current_surfaces[pixel],
            current_reservoirs[pixel],
            history_available ? &previous_camera : nullptr,
            history_available ? previous_surfaces : nullptr,
            history_available ? previous_reservoirs : nullptr,
            width, height, pixel, iteration, seed, max_history_length,
            max_candidates, normal_threshold, depth_threshold,
            destination[pixel], stats);
    std::uint32_t status_index = static_cast<std::uint32_t>(status);
    status_index = status_index < 16u ? status_index : 15u;
    if (status_output != nullptr) {
        status_output[pixel] = status_index;
    }
    atomicAdd(&counters->gi_temporal_status[status_index], 1ull);
    atomicAdd(&counters->gi_temporal_candidates,
              static_cast<unsigned long long>(stats.candidates));
    atomicAdd(&counters->gi_temporal_accepted,
              static_cast<unsigned long long>(stats.accepted));
    const std::uint32_t rejection_index =
        static_cast<std::uint32_t>(stats.rejection);
    if (rejection_index < 16u) {
        atomicAdd(&counters->gi_temporal_rejection[rejection_index], 1ull);
    }
    const std::uint32_t shift_index =
        static_cast<std::uint32_t>(stats.shift_failure);
    if (shift_index < 16u &&
        stats.shift_failure != restir::RestirGIShiftFailure::None) {
        atomicAdd(&counters->gi_shift_failures[shift_index], 1ull);
    }
}

} // namespace

void launch_restir_temporal_gi_basic(
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
    float depth_threshold, DeviceRestirCounters *counters,
    std::uint32_t block_size, std::uint32_t *status_output) {
    const std::uint32_t pixel_count = width * height;
    const std::uint32_t grid =
        (pixel_count + block_size - 1u) / block_size;
    temporal_gi_basic_kernel<<<grid, block_size>>>(
        scene, current_surfaces, current_reservoirs, previous_surfaces,
        previous_reservoirs, previous_camera, history_available,
        destination, width, height, iteration, seed, max_history_length,
        max_candidates, normal_threshold, depth_threshold, counters,
        status_output);
    RT_CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda_backend

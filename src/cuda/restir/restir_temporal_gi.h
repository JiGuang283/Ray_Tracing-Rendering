#ifndef CUDA_RESTIR_TEMPORAL_GI_H
#define CUDA_RESTIR_TEMPORAL_GI_H

#include "device_scene.h"
#include "restir_device_types.h"
#include "restir_gi_types.h"
#include "restir_surface.h"

#include <cstdint>

namespace cuda_backend {

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
    std::uint32_t block_size, std::uint32_t *status_output = nullptr);

} // namespace cuda_backend

#endif

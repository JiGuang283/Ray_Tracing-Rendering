#ifndef CUDA_RESTIR_TEMPORAL_DI_H
#define CUDA_RESTIR_TEMPORAL_DI_H

#include "device_scene.h"
#include "restir_device_types.h"
#include "restir_di_types.h"
#include "restir_surface.h"

#include <cstdint>

namespace cuda_backend {

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
    DeviceRestirCounters *counters, std::uint32_t block_size);

} // namespace cuda_backend

#endif

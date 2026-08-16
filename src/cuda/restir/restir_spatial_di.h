#ifndef CUDA_RESTIR_SPATIAL_DI_H
#define CUDA_RESTIR_SPATIAL_DI_H

#include "device_scene.h"
#include "restir_device_types.h"
#include "restir_di_types.h"
#include "restir_surface.h"

#include <cstdint>

namespace cuda_backend {

void launch_restir_spatial_di_basic(
    DeviceSceneView scene, const restir::RestirSurface *surfaces,
    const restir::RestirDIReservoir *source,
    restir::RestirDIReservoir *destination, std::uint32_t width,
    std::uint32_t height, std::uint32_t iteration,
    std::uint32_t pass_index, std::uint32_t seed,
    std::uint32_t neighbor_count, std::uint32_t max_candidates,
    float normal_threshold, float depth_threshold,
    DeviceRestirCounters *counters, std::uint32_t block_size,
    CudaRestirStatsLevel stats_level = CudaRestirStatsLevel::Full);

void launch_restir_spatial_di_pairwise(
    DeviceSceneView scene, const restir::RestirSurface *surfaces,
    const restir::RestirDIReservoir *source,
    restir::RestirDIReservoir *destination, std::uint32_t width,
    std::uint32_t height, std::uint32_t iteration,
    std::uint32_t pass_index, std::uint32_t seed,
    std::uint32_t neighbor_count, std::uint32_t max_candidates,
    float normal_threshold, float depth_threshold,
    DeviceRestirCounters *counters, std::uint32_t block_size,
    CudaRestirStatsLevel stats_level = CudaRestirStatsLevel::Full);

} // namespace cuda_backend

#endif

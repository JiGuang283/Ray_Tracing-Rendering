#ifndef CUDA_RESTIR_SPATIAL_GI_H
#define CUDA_RESTIR_SPATIAL_GI_H

#include "device_scene.h"
#include "restir_device_types.h"
#include "restir_gi_types.h"
#include "restir_surface.h"

#include <cstdint>

namespace cuda_backend {

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
    std::uint32_t block_size, std::uint32_t *status_output = nullptr,
    CudaRestirStatsLevel stats_level = CudaRestirStatsLevel::Full);

} // namespace cuda_backend

#endif

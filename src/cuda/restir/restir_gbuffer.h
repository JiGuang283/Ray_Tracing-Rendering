#ifndef CUDA_RESTIR_GBUFFER_H
#define CUDA_RESTIR_GBUFFER_H

#include "device_scene.h"
#include "restir_device_types.h"
#include "restir_surface.h"

#include <cstdint>

namespace cuda_backend {

void launch_restir_gbuffer(DeviceSceneView scene, std::uint32_t width,
                           std::uint32_t height,
                           std::uint32_t iteration,
                           std::uint32_t seed,
                           restir::RestirSurface *output,
                           DeviceRestirCounters *counters,
                           std::uint32_t block_size,
                           CudaRestirStatsLevel stats_level =
                               CudaRestirStatsLevel::Full);

} // namespace cuda_backend

#endif

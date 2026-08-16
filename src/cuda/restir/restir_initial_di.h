#ifndef CUDA_RESTIR_INITIAL_DI_H
#define CUDA_RESTIR_INITIAL_DI_H

#include "device_scene.h"
#include "restir_device_types.h"
#include "restir_di_types.h"
#include "restir_surface.h"
#include "wavefront_renderer.h"

#include <cstdint>

namespace cuda_backend {

void launch_restir_initial_di_candidates(
    DeviceSceneView scene, const restir::RestirSurface *surfaces,
    std::uint32_t width, std::uint32_t height,
    std::uint32_t iteration, std::uint32_t seed,
    std::uint32_t candidate_count,
    restir::RestirDIReservoir *reservoirs,
    DeviceRestirCounters *counters, std::uint32_t block_size,
    CudaRestirStatsLevel stats_level = CudaRestirStatsLevel::Full);

void launch_restir_initial_di_shading(
    DeviceSceneView scene, const restir::RestirSurface *surfaces,
    const restir::RestirDIReservoir *reservoirs, std::uint32_t width,
    std::uint32_t height, std::uint32_t iteration, std::uint32_t seed,
    float sample_clamp, CudaFilmPixel *film,
    DeviceRestirCounters *counters, std::uint32_t block_size,
    CudaRestirStatsLevel stats_level = CudaRestirStatsLevel::Full);

} // namespace cuda_backend

#endif

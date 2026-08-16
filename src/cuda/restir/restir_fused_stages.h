#ifndef CUDA_RESTIR_FUSED_STAGES_H
#define CUDA_RESTIR_FUSED_STAGES_H

#include "device_scene.h"
#include "restir_device_types.h"
#include "restir_di_types.h"
#include "restir_surface.h"
#include "wavefront_renderer.h"

#include <cstdint>

namespace cuda_backend {

// Fused stage entry points. Each fused kernel preserves the per-pixel
// operation order of the original separate kernels exactly, so films remain
// bit-identical while the iteration launch count decreases.

void launch_restir_fused_gbuffer_initial_di(
    DeviceSceneView scene, std::uint32_t width, std::uint32_t height,
    std::uint32_t iteration, std::uint32_t seed,
    std::uint32_t candidate_count, restir::RestirSurface *surfaces,
    restir::RestirDIReservoir *reservoirs, DeviceRestirCounters *counters,
    std::uint32_t block_size,
    CudaRestirStatsLevel stats_level = CudaRestirStatsLevel::Full);

void launch_restir_fused_initial_di_fallback_shading(
    DeviceSceneView scene, PackedTransportSettings transport,
    const restir::RestirSurface *surfaces,
    const restir::RestirDIReservoir *reservoirs, std::uint32_t width,
    std::uint32_t height, std::uint32_t iteration, std::uint32_t seed,
    float sample_clamp, CudaFilmPixel *direct_film,
    DeviceRestirCounters *counters, std::uint32_t block_size,
    CudaRestirStatsLevel stats_level = CudaRestirStatsLevel::Full);

void launch_restir_fused_fallback_reference_shading(
    DeviceSceneView scene, PackedTransportSettings transport,
    const restir::RestirSurface *surfaces, std::uint32_t width,
    std::uint32_t height, std::uint32_t iteration, std::uint32_t seed,
    float sample_clamp, CudaFilmPixel *direct_film,
    CudaFilmPixel *reference_film, DeviceRestirCounters *counters,
    std::uint32_t block_size,
    CudaRestirStatsLevel stats_level = CudaRestirStatsLevel::Full);

} // namespace cuda_backend

#endif

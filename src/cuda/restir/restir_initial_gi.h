#ifndef CUDA_RESTIR_INITIAL_GI_H
#define CUDA_RESTIR_INITIAL_GI_H

#include "device_scene.h"
#include "restir_device_types.h"
#include "restir_gi_types.h"
#include "restir_surface.h"
#include "wavefront_renderer.h"

#include <cstdint>

namespace cuda_backend {

void launch_restir_initial_gi_candidates(
    DeviceSceneView scene, const restir::RestirSurface *surfaces,
    std::uint32_t width, std::uint32_t height,
    std::uint32_t iteration, std::uint32_t seed,
    std::uint32_t candidate_count,
    const PackedTransportSettings &transport,
    restir::RestirGIReservoir *reservoirs,
    CudaFilmPixel *fallback_film,
    DeviceRestirCounters *counters, std::uint32_t block_size,
    std::uint32_t *status_output = nullptr);

void launch_restir_initial_gi_shading(
    DeviceSceneView scene, const restir::RestirSurface *surfaces,
    const restir::RestirGIReservoir *reservoirs,
    std::uint32_t width, std::uint32_t height,
    std::uint32_t iteration, std::uint32_t seed,
    CudaFilmPixel *film, DeviceRestirCounters *counters,
    std::uint32_t block_size, std::uint32_t *status_output = nullptr);

} // namespace cuda_backend

#endif

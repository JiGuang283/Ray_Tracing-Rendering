#ifndef CUDA_RESTIR_REFERENCE_SHADING_H
#define CUDA_RESTIR_REFERENCE_SHADING_H

#include "device_scene.h"
#include "restir_device_types.h"
#include "restir_surface.h"
#include "wavefront_renderer.h"

#include <cstdint>

namespace cuda_backend {

void launch_restir_reference_shading(
    DeviceSceneView scene, PackedTransportSettings transport,
    std::uint32_t width, std::uint32_t height,
    std::uint32_t iteration, std::uint32_t seed, float sample_clamp,
    CudaFilmPixel *film, DeviceRestirCounters *counters,
    std::uint32_t block_size);

void launch_restir_fallback_shading(
    DeviceSceneView scene, PackedTransportSettings transport,
    const restir::RestirSurface *surfaces, std::uint32_t width,
    std::uint32_t height, std::uint32_t iteration, std::uint32_t seed,
    CudaFilmPixel *film, DeviceRestirCounters *counters,
    std::uint32_t block_size);

} // namespace cuda_backend

#endif

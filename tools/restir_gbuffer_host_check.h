#ifndef RESTIR_GBUFFER_HOST_CHECK_H
#define RESTIR_GBUFFER_HOST_CHECK_H

#include "compiled_scene.h"
#include "restir_surface.h"

#include <cstdint>
#include <vector>

std::uint64_t compare_restir_gbuffer_host(
    const CompiledSceneView &scene, std::uint32_t width,
    std::uint32_t height, std::uint32_t iteration,
    std::uint32_t seed,
    const std::vector<restir::RestirSurface> &device_surfaces);

#endif

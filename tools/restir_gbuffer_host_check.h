#ifndef RESTIR_GBUFFER_HOST_CHECK_H
#define RESTIR_GBUFFER_HOST_CHECK_H

#include "compiled_scene.h"
#include "restir_di_types.h"
#include "restir_surface.h"
#include "wavefront_renderer.h"

#include <array>
#include <cstdint>
#include <vector>

std::uint64_t compare_restir_gbuffer_host(
    const CompiledSceneView &scene, std::uint32_t width,
    std::uint32_t height, std::uint32_t iteration,
    std::uint32_t seed,
    const std::vector<restir::RestirSurface> &device_surfaces);

struct RestirDIHostCheckResult {
    std::uint64_t reservoir_errors = 0;
    std::uint64_t direct_film_errors = 0;
    std::uint64_t initial_candidates = 0;
    std::uint64_t represented_candidates = 0;
    std::uint64_t rejected_candidates = 0;
    std::uint64_t visibility_rays = 0;
    std::array<std::uint64_t, 11> generation_status{};
    std::array<std::uint64_t, 11> shading_status{};
};

RestirDIHostCheckResult compare_restir_initial_di_host(
    const CompiledSceneView &scene, std::uint32_t width,
    std::uint32_t height, std::uint32_t iterations,
    std::uint32_t candidate_count, std::uint32_t seed,
    const std::vector<restir::RestirSurface> &device_surfaces,
    const std::vector<restir::RestirDIReservoir> &device_reservoirs,
    const std::vector<cuda_backend::CudaFilmPixel> &device_film);

#endif

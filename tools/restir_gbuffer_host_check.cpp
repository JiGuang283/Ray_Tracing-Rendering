#include "restir_gbuffer_host_check.h"

#include "restir_gbuffer_core.h"
#include "restir_di_core.h"
#include "restir_spatial_core.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace {

bool near(float left, float right, float tolerance = 1e-4f) {
    const float scale =
        std::max(1.0f, std::max(std::abs(left), std::abs(right)));
    return std::abs(left - right) <= tolerance * scale;
}

bool compare_surface(const restir::RestirSurface &device,
                     const restir::RestirSurface &host) {
    if (device.valid() != host.valid() || device.flags != host.flags) {
        return false;
    }
    if (!host.valid()) {
        return true;
    }
    if (device.instance_id != host.instance_id ||
        device.primitive_id != host.primitive_id ||
        device.material_id != host.material_id ||
        device.emitter_id != host.emitter_id ||
        !near(device.position.x, host.position.x) ||
        !near(device.position.y, host.position.y) ||
        !near(device.position.z, host.position.z) ||
        !near(device.view_depth, host.view_depth) ||
        !near(device.hit_t, host.hit_t) ||
        !near(device.barycentric_u, host.barycentric_u) ||
        !near(device.barycentric_v, host.barycentric_v) ||
        !near(device.ray_time, host.ray_time)) {
        return false;
    }
    const Float3 device_geometry =
        restir::unpack_octahedral_normal(device.geometric_normal);
    const Float3 host_geometry =
        restir::unpack_octahedral_normal(host.geometric_normal);
    const Float3 device_shading =
        restir::unpack_octahedral_normal(device.shading_normal);
    const Float3 host_shading =
        restir::unpack_octahedral_normal(host.shading_normal);
    const float geometry_dot = device_geometry.x * host_geometry.x +
                               device_geometry.y * host_geometry.y +
                               device_geometry.z * host_geometry.z;
    const float shading_dot = device_shading.x * host_shading.x +
                              device_shading.y * host_shading.y +
                              device_shading.z * host_shading.z;
    return geometry_dot >= 0.9999f && shading_dot >= 0.9999f;
}

bool compare_reservoir(const restir::RestirDIReservoir &device,
                       const restir::RestirDIReservoir &host) {
    if (device.M != host.M || device.flags != host.flags ||
        device.age != host.age) {
        return false;
    }
    if (!restir::reservoir_has_sample(host)) {
        return true;
    }
    return device.sample.light_id == host.sample.light_id &&
           device.sample.element_id == host.sample.element_id &&
           device.sample.type == host.sample.type &&
           device.sample.flags == host.sample.flags &&
           near(device.sample.canonical_data.x,
                host.sample.canonical_data.x) &&
           near(device.sample.canonical_data.y,
                host.sample.canonical_data.y) &&
           near(device.sample.canonical_data.z,
                host.sample.canonical_data.z) &&
           near(device.weight_sum, host.weight_sum) &&
           near(device.effective_M, host.effective_M) &&
           near(device.selected_target, host.selected_target) &&
           near(device.unbiased_contribution_weight,
                host.unbiased_contribution_weight);
}

} // namespace

std::uint64_t compare_restir_gbuffer_host(
    const CompiledSceneView &scene, std::uint32_t width,
    std::uint32_t height, std::uint32_t iteration,
    std::uint32_t seed,
    const std::vector<restir::RestirSurface> &device_surfaces) {
    const std::uint32_t pixel_count = width * height;
    if (device_surfaces.size() != pixel_count) {
        return pixel_count;
    }
    std::uint64_t errors = 0;
    for (std::uint32_t pixel = 0; pixel < pixel_count; ++pixel) {
        restir::RestirSurface host_surface;
        (void)restir::build_primary_surface_core(
            scene, width, height, pixel, iteration, seed, host_surface);
        if (!compare_surface(device_surfaces[pixel], host_surface)) {
            if (errors == 0u) {
                const restir::RestirSurface &device =
                    device_surfaces[pixel];
                std::cerr << "RESTIR_GBUFFER_MISMATCH pixel=" << pixel
                          << " device_valid=" << device.valid()
                          << " host_valid=" << host_surface.valid()
                          << " device_flags=" << device.flags
                          << " host_flags=" << host_surface.flags
                          << " device_instance=" << device.instance_id
                          << " host_instance=" << host_surface.instance_id
                          << " device_primitive=" << device.primitive_id
                          << " host_primitive=" << host_surface.primitive_id
                          << " device_t=" << device.hit_t
                          << " host_t=" << host_surface.hit_t
                          << " device_position=" << device.position.x << ','
                          << device.position.y << ',' << device.position.z
                          << " host_position=" << host_surface.position.x << ','
                          << host_surface.position.y << ','
                          << host_surface.position.z
                          << " device_bary=" << device.barycentric_u << ','
                          << device.barycentric_v
                          << " host_bary=" << host_surface.barycentric_u << ','
                          << host_surface.barycentric_v
                          << " device_normals=" << device.geometric_normal
                          << ',' << device.shading_normal
                          << " host_normals=" << host_surface.geometric_normal
                          << ',' << host_surface.shading_normal
                          << " device_ray_time=" << device.ray_time
                          << " host_ray_time=" << host_surface.ray_time
                          << '\n';
            }
            ++errors;
        }
    }
    return errors;
}

RestirDIHostCheckResult compare_restir_initial_di_host(
    const CompiledSceneView &scene, std::uint32_t width,
    std::uint32_t height, std::uint32_t iterations,
    std::uint32_t candidate_count, std::uint32_t seed,
    const std::vector<restir::RestirSurface> &device_surfaces,
    const std::vector<restir::RestirDIReservoir> &device_reservoirs,
    const std::vector<cuda_backend::CudaFilmPixel> &device_film) {
    RestirDIHostCheckResult result;
    const std::uint32_t pixel_count = width * height;
    if (iterations != 1u || device_surfaces.size() != pixel_count ||
        device_reservoirs.size() != pixel_count ||
        device_film.size() != pixel_count) {
        result.reservoir_errors = pixel_count;
        result.direct_film_errors = pixel_count;
        return result;
    }
    std::vector<cuda_backend::CudaFilmPixel> host_film(pixel_count);
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
        for (std::uint32_t pixel = 0; pixel < pixel_count; ++pixel) {
            const restir::RestirSurface &surface =
                device_surfaces[pixel];
            restir::RestirDIReservoir reservoir;
            restir::RestirDICandidateStats candidate_stats;
            const restir::RestirDIStatus generation =
                restir::generate_initial_di_reservoir(
                    scene, surface, width, height, pixel, iteration, seed,
                    candidate_count, reservoir, candidate_stats);
            const std::uint32_t generation_index =
                static_cast<std::uint32_t>(generation);
            if (generation_index < result.generation_status.size()) {
                ++result.generation_status[generation_index];
            }
            result.initial_candidates += candidate_stats.attempted;
            result.represented_candidates += candidate_stats.represented;
            result.rejected_candidates += candidate_stats.rejected;

            Float3 radiance{};
            std::uint32_t visibility_rays = 0;
            const restir::RestirDIStatus shading =
                restir::shade_initial_di_reservoir(
                    scene, surface, reservoir, width, height, pixel,
                    iteration, seed, radiance, visibility_rays);
            const std::uint32_t shading_index =
                static_cast<std::uint32_t>(shading);
            if (shading_index < result.shading_status.size()) {
                ++result.shading_status[shading_index];
            }
            result.visibility_rays += visibility_rays;
            host_film[pixel].radiance = packed_transport::math::add(
                host_film[pixel].radiance, radiance);
            ++host_film[pixel].sample_count;

            if (iteration + 1u == iterations &&
                !compare_reservoir(device_reservoirs[pixel], reservoir)) {
                ++result.reservoir_errors;
            }
        }
    }
    for (std::uint32_t pixel = 0; pixel < pixel_count; ++pixel) {
        if (device_film[pixel].sample_count !=
                host_film[pixel].sample_count ||
            !near(device_film[pixel].radiance.x,
                  host_film[pixel].radiance.x) ||
            !near(device_film[pixel].radiance.y,
                  host_film[pixel].radiance.y) ||
            !near(device_film[pixel].radiance.z,
                  host_film[pixel].radiance.z)) {
            ++result.direct_film_errors;
        }
    }
    return result;
}

RestirDISpatialHostCheckResult compare_restir_spatial_di_basic_host(
    const CompiledSceneView &scene, std::uint32_t width,
    std::uint32_t height, std::uint32_t iteration,
    std::uint32_t candidate_count, std::uint32_t neighbor_count,
    std::uint32_t pass_count, std::uint32_t max_candidates,
    float normal_threshold, float depth_threshold, std::uint32_t seed,
    const std::vector<restir::RestirSurface> &device_surfaces,
    const std::vector<restir::RestirDIReservoir> &device_reservoirs,
    const std::vector<cuda_backend::CudaFilmPixel> &device_film) {
    RestirDISpatialHostCheckResult result;
    const std::uint32_t pixel_count = width * height;
    if (device_surfaces.size() != pixel_count ||
        device_reservoirs.size() != pixel_count ||
        device_film.size() != pixel_count || pass_count == 0u) {
        result.reservoir_errors = pixel_count;
        result.direct_film_errors = pixel_count;
        return result;
    }

    std::vector<restir::RestirDIReservoir> source(pixel_count);
    std::vector<restir::RestirDIReservoir> destination(pixel_count);
    for (std::uint32_t pixel = 0; pixel < pixel_count; ++pixel) {
        restir::RestirDICandidateStats initial_stats;
        (void)restir::generate_initial_di_reservoir(
            scene, device_surfaces[pixel], width, height, pixel, iteration,
            seed, candidate_count, source[pixel], initial_stats);
    }
    for (std::uint32_t pass = 0; pass < pass_count; ++pass) {
        for (std::uint32_t pixel = 0; pixel < pixel_count; ++pixel) {
            restir::RestirDISpatialStats spatial_stats;
            const restir::RestirDIStatus status =
                restir::spatial_resample_di_basic(
                    scene, device_surfaces.data(), source.data(), width,
                    height, pixel, iteration, pass, seed, neighbor_count,
                    max_candidates, normal_threshold, depth_threshold,
                    destination[pixel], spatial_stats);
            ++result.spatial_status[static_cast<std::uint32_t>(status)];
            result.spatial_candidates += spatial_stats.candidates;
            result.spatial_accepted += spatial_stats.accepted;
            result.spatial_rejected += spatial_stats.rejected;
            for (std::size_t index = 0;
                 index < result.compatibility.size(); ++index) {
                result.compatibility[index] +=
                    spatial_stats.compatibility[index];
            }
        }
        source.swap(destination);
    }

    for (std::uint32_t pixel = 0; pixel < pixel_count; ++pixel) {
        if (!compare_reservoir(device_reservoirs[pixel], source[pixel])) {
            ++result.reservoir_errors;
        }
        Float3 radiance{};
        std::uint32_t visibility_rays = 0u;
        const restir::RestirDIStatus shading =
            restir::shade_initial_di_reservoir(
                scene, device_surfaces[pixel], source[pixel], width, height,
                pixel, iteration, seed, radiance, visibility_rays);
        ++result.shading_status[static_cast<std::uint32_t>(shading)];
        result.visibility_rays += visibility_rays;
        if (device_film[pixel].sample_count != 1u ||
            !near(device_film[pixel].radiance.x, radiance.x) ||
            !near(device_film[pixel].radiance.y, radiance.y) ||
            !near(device_film[pixel].radiance.z, radiance.z)) {
            ++result.direct_film_errors;
        }
    }
    return result;
}

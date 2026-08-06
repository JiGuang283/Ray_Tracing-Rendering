#include "restir_gbuffer_host_check.h"

#include "restir_gbuffer_core.h"
#include "restir_di_core.h"
#include "restir_gi_core.h"
#include "restir_gi_spatial_core.h"
#include "restir_gi_temporal_core.h"
#include "restir_spatial_core.h"
#include "restir_spatial_pairwise_core.h"
#include "restir_temporal_core.h"

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

bool compare_reservoir(const restir::RestirGIReservoir &device,
                       const restir::RestirGIReservoir &host) {
    constexpr float kReuseTolerance = 5e-4f;
    if (device.M != host.M || device.flags != host.flags ||
        device.age != host.age ||
        !near(device.weight_sum, host.weight_sum, kReuseTolerance) ||
        !near(device.effective_M, host.effective_M, kReuseTolerance) ||
        !near(device.selected_target, host.selected_target,
              kReuseTolerance) ||
        !near(device.unbiased_contribution_weight,
              host.unbiased_contribution_weight, kReuseTolerance)) {
        return false;
    }
    if (!restir::reservoir_has_sample(host)) {
        return !restir::reservoir_has_sample(device);
    }
    return device.sample.material_id == host.sample.material_id &&
           device.sample.instance_id == host.sample.instance_id &&
           device.sample.primitive_id == host.sample.primitive_id &&
           device.sample.source_pixel == host.sample.source_pixel &&
           device.sample.geometric_normal == host.sample.geometric_normal &&
           near(device.sample.position.x, host.sample.position.x,
                kReuseTolerance) &&
           near(device.sample.position.y, host.sample.position.y,
                kReuseTolerance) &&
           near(device.sample.position.z, host.sample.position.z,
                kReuseTolerance) &&
           near(device.sample.source_pdf_area,
                host.sample.source_pdf_area, kReuseTolerance) &&
           near(device.sample.suffix_radiance.x,
                host.sample.suffix_radiance.x, kReuseTolerance) &&
           near(device.sample.suffix_radiance.y,
                host.sample.suffix_radiance.y, kReuseTolerance) &&
           near(device.sample.suffix_radiance.z,
                host.sample.suffix_radiance.z, kReuseTolerance);
}

template <std::size_t Size, typename Enum>
void increment(std::array<std::uint64_t, Size> &buckets,
               Enum value) {
    const std::size_t index = static_cast<std::size_t>(value);
    if (index < buckets.size()) {
        ++buckets[index];
    }
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

static RestirDISpatialHostCheckResult compare_restir_spatial_di_host_impl(
    const CompiledSceneView &scene, std::uint32_t width,
    std::uint32_t height, std::uint32_t iteration,
    std::uint32_t candidate_count, std::uint32_t neighbor_count,
    std::uint32_t pass_count, std::uint32_t max_candidates,
    float normal_threshold, float depth_threshold, std::uint32_t seed,
    const std::vector<restir::RestirSurface> &device_surfaces,
    const std::vector<restir::RestirDIReservoir> &device_reservoirs,
    const std::vector<cuda_backend::CudaFilmPixel> &device_film,
    bool pairwise) {
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
                pairwise
                    ? restir::spatial_resample_di_pairwise(
                          scene, device_surfaces.data(), source.data(), width,
                          height, pixel, iteration, pass, seed,
                          neighbor_count, max_candidates, normal_threshold,
                          depth_threshold, destination[pixel], spatial_stats)
                    : restir::spatial_resample_di_basic(
                          scene, device_surfaces.data(), source.data(), width,
                          height, pixel, iteration, pass, seed,
                          neighbor_count, max_candidates, normal_threshold,
                          depth_threshold, destination[pixel], spatial_stats);
            ++result.spatial_status[static_cast<std::uint32_t>(status)];
            result.spatial_candidates += spatial_stats.candidates;
            result.spatial_accepted += spatial_stats.accepted;
            result.spatial_rejected += spatial_stats.rejected;
            result.pairwise_fallbacks += spatial_stats.pairwise_fallbacks;
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

RestirDISpatialHostCheckResult compare_restir_spatial_di_basic_host(
    const CompiledSceneView &scene, std::uint32_t width,
    std::uint32_t height, std::uint32_t iteration,
    std::uint32_t candidate_count, std::uint32_t neighbor_count,
    std::uint32_t pass_count, std::uint32_t max_candidates,
    float normal_threshold, float depth_threshold, std::uint32_t seed,
    const std::vector<restir::RestirSurface> &device_surfaces,
    const std::vector<restir::RestirDIReservoir> &device_reservoirs,
    const std::vector<cuda_backend::CudaFilmPixel> &device_film) {
    return compare_restir_spatial_di_host_impl(
        scene, width, height, iteration, candidate_count, neighbor_count,
        pass_count, max_candidates, normal_threshold, depth_threshold, seed,
        device_surfaces, device_reservoirs, device_film, false);
}

RestirDISpatialHostCheckResult compare_restir_spatial_di_pairwise_host(
    const CompiledSceneView &scene, std::uint32_t width,
    std::uint32_t height, std::uint32_t iteration,
    std::uint32_t candidate_count, std::uint32_t neighbor_count,
    std::uint32_t pass_count, std::uint32_t max_candidates,
    float normal_threshold, float depth_threshold, std::uint32_t seed,
    const std::vector<restir::RestirSurface> &device_surfaces,
    const std::vector<restir::RestirDIReservoir> &device_reservoirs,
    const std::vector<cuda_backend::CudaFilmPixel> &device_film) {
    return compare_restir_spatial_di_host_impl(
        scene, width, height, iteration, candidate_count, neighbor_count,
        pass_count, max_candidates, normal_threshold, depth_threshold, seed,
        device_surfaces, device_reservoirs, device_film, true);
}

RestirDITemporalHostCheckResult compare_restir_temporal_di_host(
    const CompiledSceneView &scene, const PackedCamera &previous_camera,
    std::uint32_t width, std::uint32_t height,
    std::uint32_t iteration, std::uint32_t candidate_count,
    std::uint32_t max_history_length, std::uint32_t max_candidates,
    float normal_threshold, float depth_threshold, std::uint32_t seed,
    bool pairwise,
    const std::vector<restir::RestirSurface> &previous_surfaces,
    const std::vector<restir::RestirDIReservoir> &previous_reservoirs,
    const std::vector<restir::RestirSurface> &device_current_surfaces,
    const std::vector<restir::RestirDIReservoir> &device_reservoirs,
    const std::vector<cuda_backend::CudaFilmPixel> &device_film) {
    RestirDITemporalHostCheckResult result;
    const std::uint32_t pixel_count = width * height;
    if (previous_surfaces.size() != pixel_count ||
        previous_reservoirs.size() != pixel_count ||
        device_current_surfaces.size() != pixel_count ||
        device_reservoirs.size() != pixel_count ||
        device_film.size() != pixel_count) {
        result.reservoir_errors = pixel_count;
        result.motion_errors = pixel_count;
        result.direct_film_errors = pixel_count;
        return result;
    }
    for (std::uint32_t pixel = 0u; pixel < pixel_count; ++pixel) {
        restir::RestirSurface current = device_current_surfaces[pixel];
        current.motion = kInvalidPackedIndex;
        restir::RestirDIReservoir initial;
        restir::RestirDICandidateStats initial_stats;
        (void)restir::generate_initial_di_reservoir(
            scene, current, width, height, pixel, iteration, seed,
            candidate_count, initial, initial_stats);
        restir::RestirDIReservoir temporal;
        restir::RestirDITemporalStats temporal_stats;
        const restir::RestirDIStatus status =
            pairwise
                ? restir::temporal_resample_di_pairwise(
                      scene, current, initial, &previous_camera,
                      previous_surfaces.data(), previous_reservoirs.data(),
                      width, height, pixel, iteration, seed,
                      max_history_length, max_candidates, normal_threshold,
                      depth_threshold, temporal, temporal_stats)
                : restir::temporal_resample_di_basic(
                      scene, current, initial, &previous_camera,
                      previous_surfaces.data(), previous_reservoirs.data(),
                      width, height, pixel, iteration, seed,
                      max_history_length, max_candidates, normal_threshold,
                      depth_threshold, temporal, temporal_stats);
        ++result.temporal_status[static_cast<std::uint32_t>(status)];
        ++result.rejection[
            static_cast<std::uint32_t>(temporal_stats.rejection)];
        result.temporal_candidates += temporal_stats.candidates;
        result.temporal_accepted += temporal_stats.accepted;
        result.pairwise_fallbacks += temporal_stats.pairwise_fallbacks;
        if (current.motion != device_current_surfaces[pixel].motion) {
            ++result.motion_errors;
        }
        if (!compare_reservoir(device_reservoirs[pixel], temporal)) {
            ++result.reservoir_errors;
        }

        Float3 radiance{};
        std::uint32_t visibility_rays = 0u;
        const restir::RestirDIStatus shading =
            restir::shade_initial_di_reservoir(
                scene, current, temporal, width, height, pixel, iteration,
                seed, radiance, visibility_rays);
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

RestirGIReuseHostCheckResult compare_restir_gi_reuse_host(
    const CompiledSceneView &scene, const PackedCamera *previous_camera,
    std::uint32_t width, std::uint32_t height,
    std::uint32_t iteration, std::uint32_t candidate_count,
    std::uint32_t neighbor_count, std::uint32_t pass_count,
    std::uint32_t max_history_length, std::uint32_t max_candidates,
    float normal_threshold, float depth_threshold, std::uint32_t seed,
    const PackedTransportSettings &transport, bool temporal_reuse,
    bool spatial_reuse,
    const std::vector<restir::RestirSurface> &previous_surfaces,
    const std::vector<restir::RestirGIReservoir> &previous_reservoirs,
    const std::vector<restir::RestirSurface> &device_current_surfaces,
    const std::vector<restir::RestirGIReservoir> &device_reservoirs,
    const std::vector<cuda_backend::CudaFilmPixel> &device_film) {
    RestirGIReuseHostCheckResult result;
    const std::uint32_t pixel_count = width * height;
    const bool valid_history = !temporal_reuse ||
        (previous_camera != nullptr &&
         previous_surfaces.size() == pixel_count &&
         previous_reservoirs.size() == pixel_count);
    if (!valid_history || device_current_surfaces.size() != pixel_count ||
        device_reservoirs.size() != pixel_count ||
        device_film.size() != pixel_count ||
        (spatial_reuse && pass_count == 0u)) {
        result.reservoir_errors = pixel_count;
        result.motion_errors = pixel_count;
        result.indirect_film_errors = pixel_count;
        return result;
    }

    std::vector<restir::RestirSurface> current_surfaces =
        device_current_surfaces;
    std::vector<restir::RestirGIReservoir> source(pixel_count);
    std::vector<restir::RestirGIReservoir> destination(pixel_count);
    std::vector<Float3> fallback(pixel_count);
    for (std::uint32_t pixel = 0u; pixel < pixel_count; ++pixel) {
        current_surfaces[pixel].motion = kInvalidPackedIndex;
        restir::RestirGICandidateStats stats;
        const restir::RestirGIStatus status =
            restir::generate_initial_gi_reservoir(
                scene, current_surfaces[pixel], width, height, pixel,
                iteration, seed, candidate_count, transport,
                source[pixel], stats, &fallback[pixel]);
        increment(result.generation_status, status);
        result.initial_candidates += stats.attempted;
        result.represented_candidates += stats.represented;
        result.rejected_candidates += stats.rejected;
    }

    if (temporal_reuse) {
        for (std::uint32_t pixel = 0u; pixel < pixel_count; ++pixel) {
            restir::RestirGITemporalStats stats;
            const restir::RestirGIStatus status =
                restir::temporal_resample_gi_basic(
                    scene, current_surfaces[pixel], source[pixel],
                    previous_camera, previous_surfaces.data(),
                    previous_reservoirs.data(), width, height, pixel,
                    iteration, seed, max_history_length, max_candidates,
                    normal_threshold, depth_threshold, destination[pixel],
                    stats);
            increment(result.temporal_status, status);
            increment(result.rejection, stats.rejection);
            if (stats.shift_failure != restir::RestirGIShiftFailure::None) {
                increment(result.shift_failures, stats.shift_failure);
            }
            result.temporal_candidates += stats.candidates;
            result.temporal_accepted += stats.accepted;
        }
        source.swap(destination);
    }

    if (spatial_reuse) {
        for (std::uint32_t pass = 0u; pass < pass_count; ++pass) {
            for (std::uint32_t pixel = 0u; pixel < pixel_count; ++pixel) {
                restir::RestirGISpatialStats stats;
                const restir::RestirGIStatus status =
                    restir::spatial_resample_gi_basic(
                        scene, current_surfaces.data(), source.data(), width,
                        height, pixel, iteration, pass, seed,
                        neighbor_count, max_candidates, normal_threshold,
                        depth_threshold, destination[pixel], stats);
                increment(result.spatial_status, status);
                result.spatial_candidates += stats.candidates;
                result.spatial_accepted += stats.accepted;
                result.spatial_rejected += stats.rejected;
                for (std::size_t index = 0u;
                     index < result.compatibility.size(); ++index) {
                    result.compatibility[index] +=
                        stats.compatibility[index];
                }
                for (std::size_t index = 0u;
                     index < result.shift_failures.size(); ++index) {
                    result.shift_failures[index] +=
                        stats.shift_failures[index];
                }
            }
            source.swap(destination);
        }
    }

    for (std::uint32_t pixel = 0u; pixel < pixel_count; ++pixel) {
        if (current_surfaces[pixel].motion !=
            device_current_surfaces[pixel].motion) {
            ++result.motion_errors;
        }
        if (!compare_reservoir(device_reservoirs[pixel], source[pixel])) {
            if (result.reservoir_errors == 0u) {
                const restir::RestirGIReservoir &device =
                    device_reservoirs[pixel];
                const restir::RestirGIReservoir &host = source[pixel];
                std::cerr
                    << "RESTIR_GI_RESERVOIR_MISMATCH pixel=" << pixel
                    << " device_sample=" << device.sample.source_pixel
                    << ':' << device.sample.instance_id << ':'
                    << device.sample.primitive_id
                    << " host_sample=" << host.sample.source_pixel << ':'
                    << host.sample.instance_id << ':'
                    << host.sample.primitive_id
                    << " device_M=" << device.M
                    << " host_M=" << host.M
                    << " device_effective_M=" << device.effective_M
                    << " host_effective_M=" << host.effective_M
                    << " device_weight_sum=" << device.weight_sum
                    << " host_weight_sum=" << host.weight_sum
                    << " device_target=" << device.selected_target
                    << " host_target=" << host.selected_target
                    << " device_ucw="
                    << device.unbiased_contribution_weight
                    << " host_ucw=" << host.unbiased_contribution_weight
                    << " device_age=" << device.age
                    << " host_age=" << host.age << '\n';
            }
            ++result.reservoir_errors;
        }
        Float3 radiance{};
        std::uint32_t visibility_rays = 0u;
        restir::RestirGIShiftFailure failure =
            restir::RestirGIShiftFailure::None;
        const restir::RestirGIStatus status = restir::shade_gi_reservoir(
            scene, current_surfaces[pixel], source[pixel], width, height,
            pixel, iteration, seed, radiance, visibility_rays, failure);
        radiance = packed_transport::math::add(radiance, fallback[pixel]);
        increment(result.shading_status, status);
        if (failure != restir::RestirGIShiftFailure::None) {
            increment(result.shift_failures, failure);
        }
        result.visibility_rays += visibility_rays;
        const cuda_backend::CudaFilmPixel &device = device_film[pixel];
        if (device.sample_count != 1u ||
            !near(device.radiance.x, radiance.x, 2e-4f) ||
            !near(device.radiance.y, radiance.y, 2e-4f) ||
            !near(device.radiance.z, radiance.z, 2e-4f)) {
            ++result.indirect_film_errors;
        }
    }
    return result;
}

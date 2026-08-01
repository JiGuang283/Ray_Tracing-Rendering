#ifndef PACKED_TRANSPORT_CORE_H
#define PACKED_TRANSPORT_CORE_H

#include "flat_intersector_core.h"
#include "packed_bsdf_core.h"
#include "packed_light_core.h"
#include "packed_material_core.h"
#include "surface_reconstruction_core.h"

#include <cfloat>
#include <cmath>
#include <cstdint>

namespace packed_transport {

namespace math {

RT_HOST_DEVICE RT_FORCE_INLINE float minimum(float a, float b) {
    return a < b ? a : b;
}

RT_HOST_DEVICE RT_FORCE_INLINE float maximum(float a, float b) {
    return a > b ? a : b;
}

RT_HOST_DEVICE RT_FORCE_INLINE float clamp(float value, float lower,
                                           float upper) {
    return maximum(lower, minimum(value, upper));
}

RT_HOST_DEVICE RT_FORCE_INLINE bool finite(float value) {
    return packed_bsdf::math::finite(value);
}

RT_HOST_DEVICE RT_FORCE_INLINE bool finite(Float3 value) {
    return packed_bsdf::math::finite(value);
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 add(Float3 a, Float3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 multiply(Float3 value, float scalar) {
    return {value.x * scalar, value.y * scalar, value.z * scalar};
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 multiply(Float3 a, Float3 b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

RT_HOST_DEVICE RT_FORCE_INLINE float dot(Float3 a, Float3 b) {
    return packed_bsdf::math::dot(a, b);
}

RT_HOST_DEVICE RT_FORCE_INLINE float length_squared(Float3 value) {
    return dot(value, value);
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 normalize(Float3 value) {
    return packed_bsdf::math::normalize(value);
}

RT_HOST_DEVICE RT_FORCE_INLINE float maximum_component(Float3 value) {
    return maximum(value.x, maximum(value.y, value.z));
}

RT_HOST_DEVICE RT_FORCE_INLINE bool black(Float3 value) {
    return value.x <= 0.0f && value.y <= 0.0f && value.z <= 0.0f;
}

} // namespace math

RT_HOST_DEVICE RT_FORCE_INLINE float power_heuristic(float pdf_a,
                                                      float pdf_b) {
    const float a2 = pdf_a * pdf_a;
    const float b2 = pdf_b * pdf_b;
    const float denominator = a2 + b2;
    return denominator > 0.0f ? a2 / denominator : 0.0f;
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 offset_origin(
    const PackedSurfaceInteraction &surface, Float3 direction) {
    constexpr float kOffset = 1e-4f;
    const float side = math::dot(direction, surface.geometric_normal) >= 0.0f
                           ? 1.0f
                           : -1.0f;
    return math::add(
        surface.position,
        math::multiply(surface.geometric_normal, side * kOffset));
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedRay spawn_ray(
    const PackedSurfaceInteraction &surface, Float3 direction, float time,
    float t_max = FLT_MAX) {
    PackedRay ray;
    ray.origin = offset_origin(surface, direction);
    ray.direction = math::normalize(direction);
    ray.t_min = 0.001f;
    ray.t_max = t_max;
    ray.time = time;
    return ray;
}

RT_HOST_DEVICE RT_FORCE_INLINE bool transport_uses_direct_lighting(
    PackedIntegratorType integrator) {
    return integrator == PackedIntegratorType::DirectLighting ||
           integrator == PackedIntegratorType::MISPath;
}

RT_HOST_DEVICE RT_FORCE_INLINE bool transport_uses_mis(
    PackedIntegratorType integrator) {
    return integrator == PackedIntegratorType::MISPath;
}

RT_HOST_DEVICE RT_FORCE_INLINE bool transport_uses_rr(
    PackedIntegratorType integrator) {
    return integrator != PackedIntegratorType::Path;
}

RT_HOST_DEVICE RT_FORCE_INLINE bool transport_uses_environment_light(
    PackedIntegratorType integrator) {
    return integrator == PackedIntegratorType::DirectLighting ||
           integrator == PackedIntegratorType::MISPath;
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedTransportStatus visibility(
    const CompiledSceneView &scene,
    const PackedSurfaceInteraction &surface, const PackedLightSample &sample,
    float time, RNG &rng, bool &is_visible) {
    is_visible = false;
    float maximum_distance = FLT_MAX;
    if (!sample.is_infinite() && sample.distance < FLT_MAX) {
        maximum_distance = sample.distance - 0.001f;
        if (!(maximum_distance > 0.001f)) {
            return PackedTransportStatus::Success;
        }
    }
    const PackedRay shadow =
        spawn_ray(surface, sample.wi, time, maximum_distance);
    PackedHit hit{};
    const PackedTraversalStatus status =
        packed_intersector::intersect_compiled_scene_core(scene, shadow, hit,
                                                          &rng);
    if (status == PackedTraversalStatus::StackOverflow ||
        status == PackedTraversalStatus::InvalidInput) {
        return PackedTransportStatus::TraversalFailure;
    }
    is_visible = status == PackedTraversalStatus::Miss;
    return PackedTransportStatus::Success;
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedTransportStatus evaluate_direct_sample(
    const CompiledSceneView &scene,
    const PackedSurfaceInteraction &surface,
    const PackedMaterialOutput &material, Float3 wo,
    const PackedLightSample &sample, float selection_probability,
    bool use_mis, float time, RNG &rng, Float3 &contribution,
    std::uint32_t &shadow_rays) {
    contribution = {};
    if (!(sample.pdf > 0.0f) || !(selection_probability > 0.0f) ||
        !math::finite(sample.pdf) ||
        !math::finite(selection_probability) ||
        !math::finite(sample.radiance) || !math::finite(sample.wi) ||
        math::black(sample.radiance)) {
        return PackedTransportStatus::Success;
    }
    Float3 f{};
    const PackedBSDFStatus eval_status =
        packed_bsdf::eval_packed_bsdf_core(material, wo, sample.wi, f);
    if (eval_status != PackedBSDFStatus::Success) {
        return eval_status == PackedBSDFStatus::Empty
                   ? PackedTransportStatus::Success
                   : PackedTransportStatus::BSDFFailure;
    }
    const float cosine = packed_bsdf::abs_cos_theta(material, sample.wi);
    if (math::black(f) || !(cosine > 0.0f) || !math::finite(cosine)) {
        return PackedTransportStatus::Success;
    }
    ++shadow_rays;
    bool is_visible = false;
    const PackedTransportStatus visibility_status = visibility(
        scene, surface, sample, time, rng, is_visible);
    if (visibility_status != PackedTransportStatus::Success || !is_visible) {
        return visibility_status;
    }
    const float light_pdf = sample.pdf * selection_probability;
    if (!(light_pdf > 0.0f) || !math::finite(light_pdf)) {
        return PackedTransportStatus::LightFailure;
    }
    float weight = 1.0f;
    if (!sample.is_delta() && use_mis && sample.is_bsdf_hittable()) {
        float bsdf_pdf = 0.0f;
        const PackedBSDFStatus pdf_status =
            packed_bsdf::pdf_packed_bsdf_core(material, wo, sample.wi,
                                               bsdf_pdf);
        if (pdf_status != PackedBSDFStatus::Success) {
            return PackedTransportStatus::BSDFFailure;
        }
        weight = power_heuristic(light_pdf, bsdf_pdf);
    }
    contribution = math::multiply(
        math::multiply(f, sample.radiance), cosine * weight / light_pdf);
    return math::finite(contribution) ? PackedTransportStatus::Success
                                      : PackedTransportStatus::NonFinite;
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedTransportStatus sample_direct_lighting(
    const CompiledSceneView &scene,
    const PackedSurfaceInteraction &surface,
    const PackedMaterialOutput &material, Float3 wo, bool use_mis,
    float time, RNG &rng, Float3 &radiance, std::uint32_t &shadow_rays) {
    radiance = {};
    if (material.closure_count == 0) {
        return PackedTransportStatus::Success;
    }
    for (std::uint32_t index = 0; index < scene.delta_light_indices.count;
         ++index) {
        const std::uint32_t light_id = scene.delta_light_indices[index];
        PackedLightSample sample{};
        const PackedLightStatus light_status =
            packed_light::sample_packed_light_core(scene, light_id,
                                                   surface.position, {},
                                                   sample);
        if (light_status == PackedLightStatus::NoSample) {
            continue;
        }
        if (light_status != PackedLightStatus::Success) {
            return PackedTransportStatus::LightFailure;
        }
        Float3 contribution{};
        const PackedTransportStatus status = evaluate_direct_sample(
            scene, surface, material, wo, sample, 1.0f, use_mis, time, rng,
            contribution, shadow_rays);
        if (status != PackedTransportStatus::Success) {
            return status;
        }
        radiance = math::add(radiance, contribution);
    }
    if (scene.non_delta_light_indices.count != 0) {
        SelectedPackedLightSample selected{};
        const PackedLightStatus light_status =
            packed_light::sample_non_delta_light_core(
                scene, surface.position, rng, selected);
        if (light_status != PackedLightStatus::NoSample &&
            light_status != PackedLightStatus::Success) {
            return PackedTransportStatus::LightFailure;
        }
        if (light_status == PackedLightStatus::Success) {
            Float3 contribution{};
            const PackedTransportStatus status = evaluate_direct_sample(
                scene, surface, material, wo, selected.sample,
                selected.selection_probability, use_mis, time, rng,
                contribution, shadow_rays);
            if (status != PackedTransportStatus::Success) {
                return status;
            }
            radiance = math::add(radiance, contribution);
        }
    }
    return math::finite(radiance) ? PackedTransportStatus::Success
                                  : PackedTransportStatus::NonFinite;
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedTransportStatus miss_radiance(
    const CompiledSceneView &scene, const PackedRay &ray,
    PackedIntegratorType integrator, std::uint32_t depth,
    bool delta_bounce, float previous_bsdf_pdf, Float3 &radiance) {
    radiance = scene.background;
    if (!transport_uses_environment_light(integrator)) {
        return PackedTransportStatus::Success;
    }
    const std::uint32_t environment =
        packed_light::environment_light_id(scene);
    if (environment == kInvalidPackedIndex) {
        return PackedTransportStatus::Success;
    }
    const PackedLightStatus environment_status =
        packed_light::environment_radiance_core(
            scene, scene.lights[environment], ray.direction, radiance);
    if (environment_status != PackedLightStatus::Success) {
        return environment_status == PackedLightStatus::NoSample
                   ? PackedTransportStatus::Success
                   : PackedTransportStatus::LightFailure;
    }
    if (integrator == PackedIntegratorType::MISPath && depth != 0 &&
        !delta_bounce) {
        float light_pdf = 0.0f;
        const PackedLightStatus pdf_status =
            packed_light::packed_light_pdf_core(
                scene, environment, ray.origin, ray.direction, light_pdf);
        if (pdf_status != PackedLightStatus::Success) {
            return PackedTransportStatus::LightFailure;
        }
        light_pdf *= scene.lights[environment].selection_probability;
        radiance = math::multiply(
            radiance, power_heuristic(previous_bsdf_pdf, light_pdf));
    }
    return math::finite(radiance) ? PackedTransportStatus::Success
                                  : PackedTransportStatus::NonFinite;
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedTransportStatus emitted_radiance(
    const CompiledSceneView &scene, const PackedMaterialOutput &material,
    const PackedSurfaceInteraction &surface, const PackedRay &ray,
    const PackedHit &hit, PackedIntegratorType integrator,
    std::uint32_t depth, bool delta_bounce, float previous_bsdf_pdf,
    Float3 &radiance) {
    radiance = material.emission;
    if (math::black(radiance)) {
        return PackedTransportStatus::Success;
    }
    if (integrator == PackedIntegratorType::DirectLighting && depth != 0 &&
        !delta_bounce) {
        radiance = {};
        return PackedTransportStatus::Success;
    }
    if (integrator != PackedIntegratorType::MISPath || depth == 0 ||
        delta_bounce || surface.emitter_id == kInvalidPackedIndex) {
        return PackedTransportStatus::Success;
    }
    float light_pdf = 0.0f;
    const PackedLightStatus pdf_status =
        packed_light::emitter_hit_mis_pdf_core(
            scene, surface.emitter_id, ray.origin, ray.direction, hit,
            light_pdf);
    if (pdf_status != PackedLightStatus::Success) {
        return PackedTransportStatus::LightFailure;
    }
    radiance = math::multiply(
        radiance, power_heuristic(previous_bsdf_pdf, light_pdf));
    return math::finite(radiance) ? PackedTransportStatus::Success
                                  : PackedTransportStatus::NonFinite;
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedTransportResult trace_packed_path_core(
    const CompiledSceneView &scene, PackedRay ray,
    const PackedTransportSettings &settings, RNG &rng) {
    PackedTransportResult result{};
    if (scene.aggregates.count == 0 || settings.max_depth == 0 ||
        static_cast<std::uint32_t>(settings.integrator) >
            static_cast<std::uint32_t>(PackedIntegratorType::MISPath) ||
        !math::finite(ray.origin) || !math::finite(ray.direction) ||
        math::length_squared(ray.direction) == 0.0f ||
        ray.t_max < ray.t_min) {
        result.status = PackedTransportStatus::InvalidInput;
        return result;
    }
    ray.direction = math::normalize(ray.direction);
    Float3 throughput{1.0f, 1.0f, 1.0f};
    float eta_scale = 1.0f;
    float previous_bsdf_pdf = 0.0f;
    bool delta_bounce = false;

    for (std::uint32_t depth = 0; depth < settings.max_depth; ++depth) {
        result.depth = depth + 1;
        ++result.traversal_steps;
        PackedHit hit{};
        const PackedTraversalStatus traversal =
            packed_intersector::intersect_compiled_scene_core(scene, ray, hit,
                                                              &rng);
        if (traversal == PackedTraversalStatus::Miss) {
            Float3 miss{};
            result.status = miss_radiance(
                scene, ray, settings.integrator, depth, delta_bounce,
                previous_bsdf_pdf, miss);
            if (result.status == PackedTransportStatus::Success) {
                result.radiance = math::add(
                    result.radiance, math::multiply(throughput, miss));
            }
            break;
        }
        if (traversal != PackedTraversalStatus::Hit) {
            result.status = PackedTransportStatus::TraversalFailure;
            break;
        }

        PackedSurfaceInteraction surface{};
        const PackedShadingStatus reconstruction =
            packed_reconstruction::reconstruct_compiled_hit_core(
                scene, ray, hit, surface);
        if (reconstruction != PackedShadingStatus::Success) {
            result.status = PackedTransportStatus::ReconstructionFailure;
            break;
        }
        PackedMaterialOutput material{};
        const PackedShadingStatus material_status =
            packed_material::evaluate_packed_material_core(
                scene, surface.material_id, surface, material);
        if (material_status != PackedShadingStatus::Success) {
            result.status = PackedTransportStatus::MaterialFailure;
            break;
        }
        const Float3 wo = math::normalize(
            math::multiply(ray.direction, -1.0f));
        Float3 emission{};
        result.status = emitted_radiance(
            scene, material, surface, ray, hit, settings.integrator, depth,
            delta_bounce, previous_bsdf_pdf, emission);
        if (result.status != PackedTransportStatus::Success) {
            break;
        }
        result.radiance = math::add(
            result.radiance, math::multiply(throughput, emission));

        if (transport_uses_direct_lighting(settings.integrator) &&
            material.closure_count != 0 && scene.lights.count != 0) {
            Float3 direct{};
            result.status = sample_direct_lighting(
                scene, surface, material, wo,
                transport_uses_mis(settings.integrator), ray.time, rng,
                direct, result.shadow_rays);
            if (result.status != PackedTransportStatus::Success) {
                break;
            }
            result.radiance = math::add(
                result.radiance,
                math::multiply(throughput, direct));
        }

        PackedBSDFSample sample{};
        const PackedBSDFStatus bsdf_status =
            packed_bsdf::sample_packed_bsdf_core(material, wo, rng, sample);
        if (bsdf_status == PackedBSDFStatus::Empty ||
            bsdf_status == PackedBSDFStatus::NoSample) {
            result.status = PackedTransportStatus::Success;
            break;
        }
        if (bsdf_status != PackedBSDFStatus::Success ||
            !(sample.pdf >= 1e-8f)) {
            result.status = bsdf_status == PackedBSDFStatus::Success
                                ? PackedTransportStatus::Success
                                : PackedTransportStatus::BSDFFailure;
            break;
        }
        delta_bounce = sample.is_delta();
        previous_bsdf_pdf = delta_bounce ? 0.0f : sample.pdf;
        const float cosine = packed_bsdf::abs_cos_theta(material, sample.wi);
        throughput = math::multiply(
            throughput,
            math::multiply(sample.f, cosine / sample.pdf));
        if (sample.is_transmission()) {
            eta_scale *= sample.eta * sample.eta;
        }
        if (!math::finite(throughput) || !math::finite(eta_scale)) {
            result.status = PackedTransportStatus::NonFinite;
            break;
        }
        ray = spawn_ray(surface, sample.wi, ray.time);

        if (transport_uses_rr(settings.integrator) &&
            depth >= settings.rr_start_depth) {
            const Float3 compensated = math::multiply(throughput, eta_scale);
            const float minimum_probability =
                settings.integrator == PackedIntegratorType::RussianRoulette
                    ? 0.005f
                    : 0.05f;
            const float survival = math::clamp(
                math::maximum_component(compensated), minimum_probability,
                0.95f);
            if (static_cast<float>(rng.next()) > survival) {
                result.status = PackedTransportStatus::Success;
                break;
            }
            throughput = math::multiply(throughput, 1.0f / survival);
        }
    }
    if (!math::finite(result.radiance)) {
        result.radiance = {};
        result.status = PackedTransportStatus::NonFinite;
    }
    return result;
}

RT_HOST_DEVICE RT_FORCE_INLINE Float2 random_in_unit_disk(RNG &rng) {
    for (;;) {
        const float x = 2.0f * static_cast<float>(rng.next()) - 1.0f;
        const float y = 2.0f * static_cast<float>(rng.next()) - 1.0f;
        if (x * x + y * y < 1.0f) {
            return {x, y};
        }
    }
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedRay generate_packed_camera_ray_core(
    const PackedCamera &camera, std::uint32_t pixel_x,
    std::uint32_t pixel_y, std::uint32_t width, std::uint32_t height,
    RNG &rng) {
    const float jitter_x = static_cast<float>(rng.next());
    const float jitter_y = static_cast<float>(rng.next());
    const float denominator_x = width > 1 ? static_cast<float>(width - 1)
                                          : 1.0f;
    const float denominator_y = height > 1 ? static_cast<float>(height - 1)
                                           : 1.0f;
    const float u = (static_cast<float>(pixel_x) + jitter_x) /
                    denominator_x;
    const float v = (static_cast<float>(pixel_y) + jitter_y) /
                    denominator_y;
    const Float2 disk = random_in_unit_disk(rng);
    const Float3 camera_u = math::normalize(camera.horizontal);
    const Float3 camera_v = math::normalize(camera.vertical);
    const Float3 offset = math::add(
        math::multiply(camera_u, camera.lens_radius * disk.x),
        math::multiply(camera_v, camera.lens_radius * disk.y));
    const float time_random = static_cast<float>(rng.next());
    const float time = camera.time0 +
                       time_random * (camera.time1 - camera.time0);
    PackedRay ray;
    ray.origin = math::add(camera.origin, offset);
    ray.direction = math::normalize(math::add(
        math::add(
            math::add(camera.lower_left_corner,
                      math::multiply(camera.horizontal, u)),
            math::multiply(camera.vertical, v)),
        math::multiply(math::add(camera.origin, offset), -1.0f)));
    ray.t_min = 0.001f;
    ray.t_max = FLT_MAX;
    ray.time = time;
    return ray;
}

} // namespace packed_transport

#endif

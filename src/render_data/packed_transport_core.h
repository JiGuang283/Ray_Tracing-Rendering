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
    const IntegratorPolicy &policy, std::uint32_t depth,
    bool delta_bounce, float previous_bsdf_pdf, Float3 &radiance) {
    radiance = scene.background;
    if (!policy.uses_direct_lighting()) {
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
    if (policy.uses_mis() && depth != 0 && !delta_bounce) {
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
    const PackedHit &hit, const IntegratorPolicy &policy,
    std::uint32_t depth, bool delta_bounce, float previous_bsdf_pdf,
    Float3 &radiance) {
    radiance = material.emission;
    if (math::black(radiance)) {
        return PackedTransportStatus::Success;
    }
    if (policy.uses_direct_lighting() && !policy.uses_mis() && depth != 0 &&
        !delta_bounce) {
        radiance = {};
        return PackedTransportStatus::Success;
    }
    if (!policy.uses_mis() || depth == 0 ||
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

RT_HOST_DEVICE RT_FORCE_INLINE bool valid_transport_input(
    const CompiledSceneView &scene, const PackedRay &ray,
    const PackedTransportSettings &settings) {
    return scene.aggregates.count != 0 && settings.max_depth != 0 &&
           valid_integrator_policy(settings.policy) &&
           math::finite(ray.origin) && math::finite(ray.direction) &&
           math::length_squared(ray.direction) != 0.0f &&
           ray.t_max >= ray.t_min;
}

RT_HOST_DEVICE RT_FORCE_INLINE void initialize_packed_path_state(
    const CompiledSceneView &scene, PackedRay ray,
    const PackedTransportSettings &settings, std::uint32_t rng_state,
    std::uint32_t pixel_index, std::uint32_t sample_index,
    PackedPathState &state) {
    state = {};
    state.ray = ray;
    state.throughput = {1.0f, 1.0f, 1.0f};
    state.eta_scale = 1.0f;
    state.rng_state = rng_state == 0 ? 1 : rng_state;
    state.pixel_index = pixel_index;
    state.sample_index = sample_index;
    if (!valid_transport_input(scene, ray, settings)) {
        state.status = PackedTransportStatus::InvalidInput;
        return;
    }
    state.ray.direction = math::normalize(state.ray.direction);
    state.flags = PACKED_PATH_ACTIVE;
}

RT_HOST_DEVICE RT_FORCE_INLINE void finish_packed_path(
    PackedPathState &state, PackedTransportStatus status, const RNG &rng) {
    state.status = status;
    state.flags &= ~PACKED_PATH_ACTIVE;
    state.rng_state = rng.state;
    if (!math::finite(state.radiance)) {
        state.radiance = {};
        state.status = PackedTransportStatus::NonFinite;
    }
}

RT_HOST_DEVICE RT_FORCE_INLINE void advance_packed_path_core(
    const CompiledSceneView &scene,
    const PackedTransportSettings &settings, PackedPathState &state) {
    if (!state.active()) {
        return;
    }
    RNG rng(state.rng_state);
    if (state.depth >= settings.max_depth) {
        finish_packed_path(state, PackedTransportStatus::Success, rng);
        return;
    }

    const std::uint32_t depth = state.depth++;
    ++state.traversal_steps;
    PackedHit hit{};
    const PackedTraversalStatus traversal =
        packed_intersector::intersect_compiled_scene_core(
            scene, state.ray, hit, &rng);
    if (traversal == PackedTraversalStatus::Miss) {
        Float3 miss{};
        const PackedTransportStatus status = miss_radiance(
            scene, state.ray, settings.policy, depth,
            state.delta_bounce(), state.previous_bsdf_pdf, miss);
        if (status == PackedTransportStatus::Success) {
            state.radiance = math::add(
                state.radiance, math::multiply(state.throughput, miss));
        }
        finish_packed_path(state, status, rng);
        return;
    }
    if (traversal != PackedTraversalStatus::Hit) {
        finish_packed_path(state, PackedTransportStatus::TraversalFailure,
                           rng);
        return;
    }

    PackedSurfaceInteraction surface{};
    const PackedShadingStatus reconstruction =
        packed_reconstruction::reconstruct_compiled_hit_core(
            scene, state.ray, hit, surface);
    if (reconstruction != PackedShadingStatus::Success) {
        finish_packed_path(
            state, PackedTransportStatus::ReconstructionFailure, rng);
        return;
    }
    PackedMaterialOutput material{};
    const PackedShadingStatus material_status =
        packed_material::evaluate_packed_material_core(
            scene, surface.material_id, surface, material);
    if (material_status != PackedShadingStatus::Success) {
        finish_packed_path(state, PackedTransportStatus::MaterialFailure,
                           rng);
        return;
    }
    const Float3 wo =
        math::normalize(math::multiply(state.ray.direction, -1.0f));
    Float3 emission{};
    PackedTransportStatus status = emitted_radiance(
        scene, material, surface, state.ray, hit, settings.policy,
        depth, state.delta_bounce(), state.previous_bsdf_pdf, emission);
    if (status != PackedTransportStatus::Success) {
        finish_packed_path(state, status, rng);
        return;
    }
    state.radiance = math::add(
        state.radiance, math::multiply(state.throughput, emission));

    if (settings.policy.uses_direct_lighting() &&
        material.closure_count != 0 && scene.lights.count != 0) {
        Float3 direct{};
        status = sample_direct_lighting(
            scene, surface, material, wo,
            settings.policy.uses_mis(), state.ray.time, rng,
            direct, state.shadow_rays);
        if (status != PackedTransportStatus::Success) {
            finish_packed_path(state, status, rng);
            return;
        }
        state.radiance = math::add(
            state.radiance, math::multiply(state.throughput, direct));
    }

    if (depth + 1 >= settings.max_depth) {
        finish_packed_path(state, PackedTransportStatus::Success, rng);
        return;
    }

    PackedBSDFSample sample{};
    const PackedBSDFStatus bsdf_status =
        packed_bsdf::sample_packed_bsdf_core(material, wo, rng, sample);
    if (bsdf_status == PackedBSDFStatus::Empty ||
        bsdf_status == PackedBSDFStatus::NoSample) {
        finish_packed_path(state, PackedTransportStatus::Success, rng);
        return;
    }
    if (bsdf_status != PackedBSDFStatus::Success ||
        !(sample.pdf >= 1e-8f)) {
        finish_packed_path(
            state,
            bsdf_status == PackedBSDFStatus::Success
                ? PackedTransportStatus::Success
                : PackedTransportStatus::BSDFFailure,
            rng);
        return;
    }

    if (sample.is_delta()) {
        state.flags |= PACKED_PATH_DELTA_BOUNCE;
        state.previous_bsdf_pdf = 0.0f;
    } else {
        state.flags &= ~PACKED_PATH_DELTA_BOUNCE;
        state.previous_bsdf_pdf = sample.pdf;
    }
    const float cosine = packed_bsdf::abs_cos_theta(material, sample.wi);
    state.throughput = math::multiply(
        state.throughput, math::multiply(sample.f, cosine / sample.pdf));
    if (sample.is_transmission()) {
        state.eta_scale *= sample.eta * sample.eta;
    }
    if (!math::finite(state.throughput) ||
        !math::finite(state.eta_scale)) {
        finish_packed_path(state, PackedTransportStatus::NonFinite, rng);
        return;
    }
    state.ray = spawn_ray(surface, sample.wi, state.ray.time);

    if (settings.policy.uses_russian_roulette() &&
        depth >= settings.policy.rr_start_depth) {
        const Float3 compensated =
            math::multiply(state.throughput, state.eta_scale);
        const float survival = math::clamp(
            math::maximum_component(compensated),
            settings.policy.rr_min_survival,
            0.95f);
        if (static_cast<float>(rng.next()) > survival) {
            finish_packed_path(state, PackedTransportStatus::Success, rng);
            return;
        }
        state.throughput =
            math::multiply(state.throughput, 1.0f / survival);
    }

    state.rng_state = rng.state;
    if (state.depth >= settings.max_depth) {
        finish_packed_path(state, PackedTransportStatus::Success, rng);
    }
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedTransportResult packed_path_result(
    const PackedPathState &state) {
    PackedTransportResult result{};
    result.radiance = state.radiance;
    result.status = state.status;
    result.depth = state.depth;
    result.shadow_rays = state.shadow_rays;
    result.traversal_steps = state.traversal_steps;
    return result;
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedTransportResult trace_packed_path_core(
    const CompiledSceneView &scene, PackedRay ray,
    const PackedTransportSettings &settings, RNG &rng) {
    PackedPathState state{};
    initialize_packed_path_state(scene, ray, settings, rng.state, 0, 0,
                                 state);
    while (state.active()) {
        advance_packed_path_core(scene, settings, state);
    }
    rng.state = state.rng_state;
    return packed_path_result(state);
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

RT_HOST_DEVICE RT_FORCE_INLINE std::uint32_t packed_camera_sample_seed(
    std::uint32_t base_seed, std::uint32_t pixel_index,
    std::uint32_t sample_index) {
    const std::uint32_t pixel_seed = mix_seed(base_seed, pixel_index + 1u);
    return mix_seed(pixel_seed, sample_index + 1u);
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
    const Float2 disk = packed_transport::random_in_unit_disk(rng);
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

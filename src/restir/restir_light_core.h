#ifndef RESTIR_LIGHT_CORE_H
#define RESTIR_LIGHT_CORE_H

#include "packed_light_core.h"
#include "restir_di_types.h"

namespace restir {
namespace light_detail {

RT_HOST_DEVICE RT_FORCE_INLINE bool valid_barycentrics(
    Float4 value) noexcept {
    using namespace packed_light::math;
    return finite(value.x) && finite(value.y) && finite(value.z) &&
           value.x >= 0.0f && value.y >= 0.0f && value.z >= 0.0f &&
           absolute((value.x + value.y + value.z) - 1.0f) <= 1e-5f;
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedLightStatus initialize_evaluation(
    const CompiledSceneView &scene, const RestirLightSample &canonical,
    const PackedLight *&light, PackedLightSample &sample) noexcept {
    light = nullptr;
    if (!canonical.valid() || canonical.light_id >= scene.lights.count) {
        return PackedLightStatus::InvalidInput;
    }
    light = &scene.lights[canonical.light_id];
    if (canonical.type != static_cast<std::uint32_t>(light->type) ||
        (light->flags & PACKED_LIGHT_DELTA) != 0u) {
        return PackedLightStatus::InvalidInput;
    }
    return packed_light::initialize_sample(scene, canonical.light_id,
                                           sample);
}

} // namespace light_detail

RT_HOST_DEVICE RT_FORCE_INLINE PackedLightStatus
generate_restir_light_sample_core(
    const CompiledSceneView &scene, float selection_random,
    Float2 sample_random, RestirLightSample &canonical,
    float &selection_probability) noexcept {
    canonical = {};
    selection_probability = 0.0f;
    std::uint32_t light_id = kInvalidPackedIndex;
    std::uint32_t selection_index = kInvalidPackedIndex;
    const PackedLightStatus selection_status =
        packed_light::select_non_delta_light_core(
            scene, selection_random, light_id, selection_index,
            selection_probability);
    if (selection_status != PackedLightStatus::Success) {
        return selection_status;
    }
    const PackedLight &light = scene.lights[light_id];
    if ((light.flags & PACKED_LIGHT_DELTA) != 0u) {
        return PackedLightStatus::InvalidDistribution;
    }

    canonical.light_id = light_id;
    canonical.type = static_cast<std::uint32_t>(light.type);
    canonical.flags = light.flags;
    switch (light.type) {
    case PackedLightType::Quad:
        canonical.canonical_data = {
            packed_light::unit_random(sample_random.x),
            packed_light::unit_random(sample_random.y), 0.0f, 0.0f};
        return PackedLightStatus::Success;
    case PackedLightType::Environment: {
        PackedLightSample sampled{};
        const PackedLightStatus init = packed_light::initialize_sample(
            scene, light_id, sampled);
        if (init != PackedLightStatus::Success) {
            return init;
        }
        const PackedLightStatus status = packed_light::sample_environment(
            scene, light, sample_random, sampled);
        if (status != PackedLightStatus::Success) {
            return status;
        }
        canonical.element_id = sampled.element_id;
        canonical.canonical_data = {sampled.wi.x, sampled.wi.y,
                                    sampled.wi.z, 0.0f};
        return PackedLightStatus::Success;
    }
    case PackedLightType::SphereEmitter: {
        const Float3 normal =
            packed_light::sphere_sample_normal(sample_random);
        canonical.element_id = light.instance_id;
        canonical.canonical_data = {normal.x, normal.y, normal.z, 0.0f};
        return PackedLightStatus::Success;
    }
    case PackedLightType::TriangleEmitter:
    case PackedLightType::MeshEmitter: {
        std::uint32_t element_index = 0;
        float local_random = 0.0f;
        float element_probability = 0.0f;
        if (!packed_light::choose_mesh_element(
                scene, light, sample_random.x, element_index,
                local_random, element_probability)) {
            return PackedLightStatus::InvalidDistribution;
        }
        const float root = ::sqrtf(local_random);
        const float b0 = 1.0f - root;
        const float b1 = packed_light::unit_random(sample_random.y) * root;
        const float b2 = 1.0f - b0 - b1;
        canonical.element_id = scene.light_element_indices[
            light.element_indices.offset + element_index];
        canonical.canonical_data = {b0, b1, b2, 0.0f};
        return PackedLightStatus::Success;
    }
    case PackedLightType::Point:
    case PackedLightType::Directional:
    case PackedLightType::Spot:
        return PackedLightStatus::InvalidDistribution;
    }
    return PackedLightStatus::InvalidInput;
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedLightStatus
evaluate_restir_light_sample_core(
    const CompiledSceneView &scene, const RestirLightSample &canonical,
    Float3 origin, PackedLightSample &sample) noexcept {
    sample = {};
    const PackedLight *light = nullptr;
    const PackedLightStatus init = light_detail::initialize_evaluation(
        scene, canonical, light, sample);
    if (init != PackedLightStatus::Success ||
        !packed_light::math::finite(origin)) {
        return init == PackedLightStatus::Success
                   ? PackedLightStatus::InvalidInput
                   : init;
    }

    switch (light->type) {
    case PackedLightType::Quad:
        return packed_light::sample_quad(
            *light, origin,
            {canonical.canonical_data.x, canonical.canonical_data.y},
            sample);
    case PackedLightType::Environment: {
        sample.wi = {canonical.canonical_data.x,
                     canonical.canonical_data.y,
                     canonical.canonical_data.z};
        if (!packed_light::math::finite(sample.wi) ||
            !(packed_light::math::length_squared(sample.wi) > 0.0f)) {
            return PackedLightStatus::InvalidInput;
        }
        sample.distance = FLT_MAX;
        sample.element_id = canonical.element_id;
        const PackedLightStatus radiance_status =
            packed_light::environment_radiance_core(
                scene, *light, sample.wi, sample.radiance);
        if (radiance_status != PackedLightStatus::Success) {
            return radiance_status;
        }
        sample.pdf =
            packed_light::environment_pdf(scene, *light, sample.wi);
        return packed_light::finish_sample(sample);
    }
    case PackedLightType::SphereEmitter:
        return packed_light::evaluate_sphere_surface_sample(
            scene, *light, origin,
            {canonical.canonical_data.x, canonical.canonical_data.y,
             canonical.canonical_data.z},
            sample);
    case PackedLightType::TriangleEmitter:
    case PackedLightType::MeshEmitter: {
        if (!light_detail::valid_barycentrics(canonical.canonical_data)) {
            return PackedLightStatus::InvalidInput;
        }
        std::uint32_t element_index = 0;
        if (!packed_light::find_mesh_element(
                scene, *light, canonical.element_id, element_index)) {
            return PackedLightStatus::InvalidInput;
        }
        return packed_light::evaluate_mesh_element_sample(
            scene, *light, origin, element_index,
            canonical.canonical_data.x, canonical.canonical_data.y,
            canonical.canonical_data.z, sample);
    }
    case PackedLightType::Point:
    case PackedLightType::Directional:
    case PackedLightType::Spot:
        return PackedLightStatus::InvalidInput;
    }
    return PackedLightStatus::InvalidInput;
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedLightStatus
sample_restir_non_delta_light_core(
    const CompiledSceneView &scene, Float3 origin, RNG &rng,
    RestirLightSample &canonical, PackedLightSample &sample,
    float &selection_probability) noexcept {
    const float selection_random = static_cast<float>(rng.next());
    const Float2 sample_random{static_cast<float>(rng.next()),
                               static_cast<float>(rng.next())};
    const PackedLightStatus generate_status =
        generate_restir_light_sample_core(
            scene, selection_random, sample_random, canonical,
            selection_probability);
    if (generate_status != PackedLightStatus::Success) {
        sample = {};
        return generate_status;
    }
    return evaluate_restir_light_sample_core(scene, canonical, origin,
                                             sample);
}

} // namespace restir

#endif

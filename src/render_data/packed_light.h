#ifndef PACKED_LIGHT_H
#define PACKED_LIGHT_H

#include "compiled_scene.h"
#include "rng.h"

PackedLightStatus sample_packed_light(const CompiledSceneView &scene,
                                      std::uint32_t light_id, Float3 origin,
                                      Float2 random,
                                      PackedLightSample &sample);

PackedLightStatus sample_packed_non_delta_light(
    const CompiledSceneView &scene, Float3 origin, RNG &rng,
    SelectedPackedLightSample &sample);

PackedLightStatus evaluate_packed_light_pdf(
    const CompiledSceneView &scene, std::uint32_t light_id, Float3 origin,
    Float3 direction, float &pdf);

PackedLightStatus evaluate_packed_light_sampler_pdf(
    const CompiledSceneView &scene, Float3 origin, Float3 direction,
    float &pdf);

PackedLightStatus evaluate_packed_emitter_hit_mis_pdf(
    const CompiledSceneView &scene, std::uint32_t light_id, Float3 origin,
    Float3 direction, const PackedHit &hit, float &pdf);

PackedLightStatus evaluate_packed_environment(
    const CompiledSceneView &scene, std::uint32_t light_id,
    Float3 direction, Float3 &radiance);

std::uint32_t find_packed_environment_light(const CompiledSceneView &scene);

#endif

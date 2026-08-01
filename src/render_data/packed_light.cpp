#include "render_data/packed_light.h"

#include "render_data/packed_light_core.h"

PackedLightStatus sample_packed_light(const CompiledSceneView &scene,
                                      std::uint32_t light_id, Float3 origin,
                                      Float2 random,
                                      PackedLightSample &sample) {
    return packed_light::sample_packed_light_core(scene, light_id, origin,
                                                  random, sample);
}

PackedLightStatus sample_packed_non_delta_light(
    const CompiledSceneView &scene, Float3 origin, RNG &rng,
    SelectedPackedLightSample &sample) {
    return packed_light::sample_non_delta_light_core(scene, origin, rng,
                                                     sample);
}

PackedLightStatus evaluate_packed_light_pdf(
    const CompiledSceneView &scene, std::uint32_t light_id, Float3 origin,
    Float3 direction, float &pdf) {
    return packed_light::packed_light_pdf_core(scene, light_id, origin,
                                               direction, pdf);
}

PackedLightStatus evaluate_packed_light_sampler_pdf(
    const CompiledSceneView &scene, Float3 origin, Float3 direction,
    float &pdf) {
    return packed_light::light_sampler_pdf_core(scene, origin, direction,
                                                pdf);
}

PackedLightStatus evaluate_packed_emitter_hit_mis_pdf(
    const CompiledSceneView &scene, std::uint32_t light_id, Float3 origin,
    Float3 direction, const PackedHit &hit, float &pdf) {
    return packed_light::emitter_hit_mis_pdf_core(
        scene, light_id, origin, direction, hit, pdf);
}

PackedLightStatus evaluate_packed_environment(
    const CompiledSceneView &scene, std::uint32_t light_id,
    Float3 direction, Float3 &radiance) {
    if (light_id >= scene.lights.count ||
        scene.lights[light_id].type != PackedLightType::Environment) {
        radiance = {};
        return PackedLightStatus::InvalidInput;
    }
    return packed_light::environment_radiance_core(
        scene, scene.lights[light_id], direction, radiance);
}

std::uint32_t find_packed_environment_light(const CompiledSceneView &scene) {
    return packed_light::environment_light_id(scene);
}

#include "packed_material.h"

#include "packed_material_core.h"

PackedShadingStatus evaluate_packed_material_status(
    const CompiledSceneView &scene, std::uint32_t material_id,
    const PackedSurfaceInteraction &surface, PackedMaterialOutput &output,
    std::uint32_t *max_texture_stack) {
    return packed_material::evaluate_packed_material_core(
        scene, material_id, surface, output, max_texture_stack);
}

bool evaluate_packed_material(const CompiledSceneView &scene,
                              std::uint32_t material_id,
                              const PackedSurfaceInteraction &surface,
                              PackedMaterialOutput &output) {
    return evaluate_packed_material_status(scene, material_id, surface,
                                           output) ==
           PackedShadingStatus::Success;
}

bool evaluate_packed_material_emission(
    const CompiledSceneView &scene, std::uint32_t material_id,
    const PackedTextureEvalContext &context, Float3 &emission) {
    return packed_material::evaluate_packed_material_emission_core(
               scene, material_id, context, emission) ==
           PackedShadingStatus::Success;
}

#ifndef PACKED_MATERIAL_H
#define PACKED_MATERIAL_H

#include "packed_texture.h"

PackedShadingStatus evaluate_packed_material_status(
    const CompiledSceneView &scene, std::uint32_t material_id,
    const PackedSurfaceInteraction &surface, PackedMaterialOutput &output,
    std::uint32_t *max_texture_stack = nullptr);

bool evaluate_packed_material(const CompiledSceneView &scene,
                              std::uint32_t material_id,
                              const PackedSurfaceInteraction &surface,
                              PackedMaterialOutput &output);

bool evaluate_packed_material_emission(
    const CompiledSceneView &scene, std::uint32_t material_id,
    const PackedTextureEvalContext &context, Float3 &emission);

#endif

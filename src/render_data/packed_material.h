#ifndef PACKED_MATERIAL_H
#define PACKED_MATERIAL_H

#include "packed_texture.h"

bool evaluate_packed_material_emission(
    const CompiledSceneView &scene, std::uint32_t material_id,
    const PackedTextureEvalContext &context, Float3 &emission);

#endif

#include "packed_material.h"

#include <cmath>

bool evaluate_packed_material_emission(
    const CompiledSceneView &scene, std::uint32_t material_id,
    const PackedTextureEvalContext &context, Float3 &emission) {
    emission = {};
    if (material_id >= scene.materials.count) {
        return false;
    }
    const PackedMaterial &material = scene.materials[material_id];
    if ((material.flags & PACKED_MATERIAL_EMISSIVE) == 0) {
        return true;
    }

    std::uint32_t texture_id = kInvalidPackedIndex;
    float strength = 1.0f;
    if (material.type == PackedMaterialType::DiffuseLight) {
        texture_id = material.texture_ids[0];
    } else if (material.type == PackedMaterialType::Principled) {
        texture_id = material.texture_ids[4];
        strength = material.parameters[0].x;
    } else {
        emission = {material.emission_estimate.x,
                    material.emission_estimate.y,
                    material.emission_estimate.z};
        return true;
    }

    Float4 texture;
    if (!evaluate_packed_texture(scene, texture_id, context, texture)) {
        return false;
    }
    emission = {texture.x * strength, texture.y * strength,
                texture.z * strength};
    return std::isfinite(emission.x) && std::isfinite(emission.y) &&
           std::isfinite(emission.z);
}

#include "packed_texture.h"

#include "packed_texture_core.h"

PackedShadingStatus evaluate_packed_texture_status(
    const CompiledSceneView &scene, std::uint32_t texture_id,
    const PackedTextureEvalContext &context, Float4 &sample,
    std::uint32_t *max_stack_used) {
    return packed_texture::evaluate_packed_texture_core(
        scene, texture_id, context, sample, max_stack_used);
}

bool evaluate_packed_texture(const CompiledSceneView &scene,
                             std::uint32_t texture_id,
                             const PackedTextureEvalContext &context,
                             Float4 &sample) {
    return evaluate_packed_texture_status(scene, texture_id, context,
                                          sample) ==
           PackedShadingStatus::Success;
}

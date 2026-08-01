#ifndef PACKED_TEXTURE_H
#define PACKED_TEXTURE_H

#include "compiled_scene.h"

struct PackedTextureEvalContext {
    Float3 position;
    Float2 uv0;
    Float4 vertex_color{1.0f, 1.0f, 1.0f, 1.0f};
};

bool evaluate_packed_texture(const CompiledSceneView &scene,
                             std::uint32_t texture_id,
                             const PackedTextureEvalContext &context,
                             Float4 &sample);

#endif

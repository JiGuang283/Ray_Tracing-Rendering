#ifndef NORMAL_MAPPING_H
#define NORMAL_MAPPING_H

#include "texture.h"

enum class NormalMapConvention {
    OpenGL,
    DirectX
};

struct NormalMapSettings {
    NormalMapConvention convention = NormalMapConvention::OpenGL;
    double strength = 1.0;
};

ShadingFrame apply_normal_map(const ShaderEvalContext &context,
                              const TextureHandle &normal_map,
                              const NormalMapSettings &settings = {});

#endif

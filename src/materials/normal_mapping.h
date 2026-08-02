#ifndef NORMAL_MAPPING_H
#define NORMAL_MAPPING_H

#include "normal_map_types.h"
#include "texture.h"

ShadingFrame apply_normal_map(const ShaderEvalContext &context,
                              const TextureHandle &normal_map,
                              const NormalMapSettings &settings = {});

#endif

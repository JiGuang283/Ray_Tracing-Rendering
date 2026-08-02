#include "normal_mapping.h"

#include <algorithm>

ShadingFrame apply_normal_map(const ShaderEvalContext &context,
                              const TextureHandle &normal_map,
                              const NormalMapSettings &settings) {
    if (!normal_map) {
        return context.frame;
    }

    const color encoded = normal_map->evaluate(context).rgb;
    const double strength = std::max(0.0, settings.strength);
    vec3 local((2.0 * encoded.x() - 1.0) * strength,
               (2.0 * encoded.y() - 1.0) * strength,
               2.0 * encoded.z() - 1.0);
    if (settings.convention == NormalMapConvention::DirectX) {
        local[1] = -local.y();
    }
    if (local.near_zero()) {
        return context.frame;
    }

    vec3 mapped = unit_vector(context.frame.to_world(unit_vector(local)));
    if (dot(mapped, context.geometry_normal) < 0.0) {
        mapped = -mapped;
    }

    ShadingFrame frame;
    frame.build_from_tangent_space(mapped, context.frame.tangent,
                                   context.frame.bitangent);
    return frame;
}

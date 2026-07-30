#ifndef SHADER_CONTEXT_H
#define SHADER_CONTEXT_H

#include "interaction.h"

struct ShaderEvalContext {
    point3 position{0, 0, 0};
    vec2 uv0{0, 0};
    vec3 geometry_normal{0, 0, 1};
    vec3 shading_normal{0, 0, 1};
    ShadingFrame frame;
    vec3 wo{0, 0, 1};
    double time = 0.0;
    bool front_face = true;
    int primitive_id = -1;
    int material_id = -1;

    static ShaderEvalContext from_surface(const SurfaceInteraction &surface,
                                          const vec3 &wo = vec3(0, 0, 0),
                                          double time = 0.0) {
        ShaderEvalContext context;
        context.position = surface.p;
        context.uv0 = vec2(surface.u, surface.v);
        context.geometry_normal = surface.geometry_normal;
        context.shading_normal = surface.shading_normal;
        context.frame = surface.frame;
        context.wo =
            wo.near_zero() ? surface.frame.normal : unit_vector(wo);
        context.time = time;
        context.front_face = surface.front_face;
        context.primitive_id = surface.primitive_id;
        context.material_id = surface.material_id;
        return context;
    }
};

#endif

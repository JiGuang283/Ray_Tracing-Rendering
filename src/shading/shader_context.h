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
                                          double time = 0.0);
};

#endif

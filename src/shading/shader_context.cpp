#include "shader_context.h"

ShaderEvalContext
ShaderEvalContext::from_surface(const SurfaceInteraction &surface,
                                const vec3 &outgoing, double ray_time) {
    ShaderEvalContext context;
    context.position = surface.p;
    context.uv0 = vec2(surface.u, surface.v);
    context.geometry_normal = surface.geometry_normal;
    context.shading_normal = surface.shading_normal;
    context.frame = surface.frame;
    context.wo =
        outgoing.near_zero() ? surface.frame.normal : unit_vector(outgoing);
    context.time = ray_time;
    context.front_face = surface.front_face;
    context.primitive_id = surface.primitive_id;
    context.material_id = surface.material_id;
    return context;
}

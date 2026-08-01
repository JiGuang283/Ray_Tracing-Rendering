#include "material_emission.h"

#include "material.h"

color evaluate_material_emission(
    const std::shared_ptr<const MaterialInstance> &material,
    const point3 &position, const vec2 &uv, const vec3 &geometry_normal,
    const vec3 &dpdu, const vec3 &dpdv, bool front_face, int primitive_id,
    int material_id, const color &vertex_color, double vertex_alpha) {
    if (!material || !material->is_emissive()) {
        return color(0, 0, 0);
    }

    ShaderEvalContext context;
    context.position = position;
    context.uv0 = uv;
    context.geometry_normal = geometry_normal;
    context.shading_normal = geometry_normal;
    context.vertex_color = vertex_color;
    context.vertex_alpha = vertex_alpha;
    context.frame.build_from_tangent_space(geometry_normal, dpdu, dpdv);
    context.wo = geometry_normal;
    context.front_face = front_face;
    context.primitive_id = primitive_id;
    context.material_id = material_id;

    ShaderScratch scratch;
    MaterialOutput output;
    material->evaluate(context, scratch, output);
    return output.has_emission ? output.emission : color(0, 0, 0);
}

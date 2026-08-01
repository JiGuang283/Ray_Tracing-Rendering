#ifndef MATERIAL_EMISSION_H
#define MATERIAL_EMISSION_H

#include "vec3.h"

#include <memory>

class MaterialInstance;

color evaluate_material_emission(
    const std::shared_ptr<const MaterialInstance> &material,
    const point3 &position, const vec2 &uv, const vec3 &geometry_normal,
    const vec3 &dpdu, const vec3 &dpdv, bool front_face,
    int primitive_id = -1, int material_id = -1,
    const color &vertex_color = color(1, 1, 1),
    double vertex_alpha = 1.0);

#endif

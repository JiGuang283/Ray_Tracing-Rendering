#ifndef GEOMETRY_TRANSFORM_H
#define GEOMETRY_TRANSFORM_H

#include "vec3.h"

#include <cmath>

inline vec3 transform_normal_by_inverse_scale(const vec3 &normal,
                                              const vec3 &scale) {
    constexpr double epsilon = 1e-12;
    vec3 transformed(
        std::abs(scale.x()) > epsilon ? normal.x() / scale.x() : 0.0,
        std::abs(scale.y()) > epsilon ? normal.y() / scale.y() : 0.0,
        std::abs(scale.z()) > epsilon ? normal.z() / scale.z() : 0.0);
    return transformed.near_zero() ? normal : unit_vector(transformed);
}

#endif

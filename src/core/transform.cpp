#include "transform.h"

#include <cmath>

Transform::Transform() = default;

Transform::Transform(const Matrix4 &object_to_world)
    : m_object_to_world(object_to_world),
      m_world_to_object(object_to_world.inverse()) {
}

Transform::Transform(const Matrix4 &object_to_world,
                     const Matrix4 &world_to_object)
    : m_object_to_world(object_to_world),
      m_world_to_object(world_to_object) {
}

Transform Transform::translate(const vec3 &offset) {
    return Transform(Matrix4::translation(offset),
                     Matrix4::translation(-offset));
}

Transform Transform::scale(const vec3 &factors) {
    if (std::abs(factors.x()) <= 1e-15 ||
        std::abs(factors.y()) <= 1e-15 ||
        std::abs(factors.z()) <= 1e-15) {
        throw std::invalid_argument("Transform scale must be non-zero.");
    }
    return Transform(Matrix4::scale(factors),
                     Matrix4::scale(vec3(1.0 / factors.x(),
                                         1.0 / factors.y(),
                                         1.0 / factors.z())));
}

Transform Transform::rotate(const Quaternion &rotation) {
    const Matrix4 matrix = Matrix4::rotation(rotation);
    return Transform(matrix, matrix.transposed());
}

Transform Transform::rotate_y(double angle_degrees) {
    const Matrix4 matrix = Matrix4::rotation_y(angle_degrees);
    return Transform(matrix, matrix.transposed());
}

Transform Transform::from_trs(const vec3 &translation,
                              const Quaternion &rotation,
                              const vec3 &scale_factors) {
    return Transform::translate(translation) * Transform::rotate(rotation) *
           Transform::scale(scale_factors);
}

point3 Transform::point_to_world(const point3 &point) const {
    return m_object_to_world.transform_point(point);
}

point3 Transform::point_to_object(const point3 &point) const {
    return m_world_to_object.transform_point(point);
}

vec3 Transform::vector_to_world(const vec3 &vector) const {
    return m_object_to_world.transform_vector(vector);
}

vec3 Transform::vector_to_object(const vec3 &vector) const {
    return m_world_to_object.transform_vector(vector);
}

vec3 Transform::normal_to_world(const vec3 &normal) const {
    return m_world_to_object.transposed().transform_vector(normal);
}

vec3 Transform::normal_to_object(const vec3 &normal) const {
    return m_object_to_world.transposed().transform_vector(normal);
}

ray Transform::ray_to_world(const ray &value) const {
    return ray(point_to_world(value.origin()), vector_to_world(value.direction()),
               value.time());
}

ray Transform::ray_to_object(const ray &value) const {
    return ray(point_to_object(value.origin()),
               vector_to_object(value.direction()), value.time());
}

aabb Transform::bounds_to_world(const aabb &bounds) const {
    return transform_bounds(m_object_to_world, bounds);
}

aabb Transform::bounds_to_object(const aabb &bounds) const {
    return transform_bounds(m_world_to_object, bounds);
}

bool Transform::swaps_handedness() const {
    return m_object_to_world.linear_determinant() < 0.0;
}

const Matrix4 &Transform::object_to_world() const {
    return m_object_to_world;
}

const Matrix4 &Transform::world_to_object() const {
    return m_world_to_object;
}

aabb Transform::transform_bounds(const Matrix4 &matrix,
                                 const aabb &bounds) {
    point3 minimum(infinity, infinity, infinity);
    point3 maximum(-infinity, -infinity, -infinity);
    for (int x = 0; x < 2; ++x) {
        for (int y = 0; y < 2; ++y) {
            for (int z = 0; z < 2; ++z) {
                const point3 corner(
                    x == 0 ? bounds.min().x() : bounds.max().x(),
                    y == 0 ? bounds.min().y() : bounds.max().y(),
                    z == 0 ? bounds.min().z() : bounds.max().z());
                const point3 transformed = matrix.transform_point(corner);
                for (int axis = 0; axis < 3; ++axis) {
                    minimum[axis] =
                        std::min(minimum[axis], transformed[axis]);
                    maximum[axis] =
                        std::max(maximum[axis], transformed[axis]);
                }
            }
        }
    }
    return aabb(minimum, maximum);
}

Transform operator*(const Transform &left, const Transform &right) {
    return Transform(left.object_to_world() * right.object_to_world(),
                     right.world_to_object() * left.world_to_object());
}

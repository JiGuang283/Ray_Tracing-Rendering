#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "aabb.h"
#include "matrix4.h"
#include "ray.h"

class Transform {
  public:
    Transform();
    explicit Transform(const Matrix4 &object_to_world);
    Transform(const Matrix4 &object_to_world,
              const Matrix4 &world_to_object);

    static Transform translate(const vec3 &offset);
    static Transform scale(const vec3 &factors);
    static Transform rotate(const Quaternion &rotation);
    static Transform rotate_y(double angle_degrees);
    static Transform from_trs(const vec3 &translation,
                              const Quaternion &rotation,
                              const vec3 &scale);

    point3 point_to_world(const point3 &point) const;
    point3 point_to_object(const point3 &point) const;
    vec3 vector_to_world(const vec3 &vector) const;
    vec3 vector_to_object(const vec3 &vector) const;
    vec3 normal_to_world(const vec3 &normal) const;
    vec3 normal_to_object(const vec3 &normal) const;
    ray ray_to_world(const ray &value) const;
    ray ray_to_object(const ray &value) const;
    aabb bounds_to_world(const aabb &bounds) const;
    aabb bounds_to_object(const aabb &bounds) const;

    bool swaps_handedness() const;
    const Matrix4 &object_to_world() const;
    const Matrix4 &world_to_object() const;

  private:
    static aabb transform_bounds(const Matrix4 &matrix,
                                 const aabb &bounds);

    Matrix4 m_object_to_world;
    Matrix4 m_world_to_object;
};

Transform operator*(const Transform &left, const Transform &right);

#endif

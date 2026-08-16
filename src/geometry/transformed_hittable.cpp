#include "transformed_hittable.h"

#include <stdexcept>

TransformedHittable::TransformedHittable(shared_ptr<hittable> child,
                                         Transform object_to_world)
    : m_child(std::move(child)),
      m_object_to_world(std::move(object_to_world)) {
    if (!m_child) {
        throw std::invalid_argument(
            "TransformedHittable requires a child object.");
    }
}

bool TransformedHittable::hit(const ray &world_ray, double t_min,
                              double t_max, hit_record &record) const {
    const ray object_ray = m_object_to_world.ray_to_object(world_ray);
    if (!m_child->hit(object_ray, t_min, t_max, record)) {
        return false;
    }
    transform_record(world_ray, record);
    return true;
}

bool TransformedHittable::hit(const ray &world_ray, double t_min,
                              double t_max, hit_record &record,
                              RNG &rng) const {
    const ray object_ray = m_object_to_world.ray_to_object(world_ray);
    if (!m_child->hit(object_ray, t_min, t_max, record, rng)) {
        return false;
    }
    transform_record(world_ray, record);
    return true;
}

bool TransformedHittable::occluded(const ray &world_ray, double t_min,
                                   double t_max, RNG &rng) const {
    const ray object_ray = m_object_to_world.ray_to_object(world_ray);
    return m_child->occluded(object_ray, t_min, t_max, rng);
}

void TransformedHittable::transform_record(const ray &world_ray,
                                           hit_record &record) const {
    vec3 outward_geometry = record.front_face ? record.geometric_normal
                                              : -record.geometric_normal;
    vec3 outward_shading = record.front_face ? record.normal : -record.normal;
    outward_geometry =
        m_object_to_world.normal_to_world(outward_geometry);
    if (m_object_to_world.swaps_handedness()) {
        outward_geometry = -outward_geometry;
    }
    outward_geometry = unit_vector(outward_geometry);
    outward_shading = unit_vector(
        m_object_to_world.normal_to_world(outward_shading));
    if (dot(outward_shading, outward_geometry) < 0.0) {
        outward_shading = -outward_shading;
    }

    record.p = m_object_to_world.point_to_world(record.p);
    record.front_face = dot(world_ray.direction(), outward_geometry) < 0.0;
    record.geometric_normal =
        record.front_face ? outward_geometry : -outward_geometry;
    record.normal = record.front_face ? outward_shading : -outward_shading;
    record.dpdu = m_object_to_world.vector_to_world(record.dpdu);
    record.dpdv = m_object_to_world.vector_to_world(record.dpdv);
}

bool TransformedHittable::bounding_box(double time0, double time1,
                                       aabb &output_box) const {
    aabb child_bounds;
    if (!m_child->bounding_box(time0, time1, child_bounds)) {
        return false;
    }
    output_box = m_object_to_world.bounds_to_world(child_bounds);
    return true;
}

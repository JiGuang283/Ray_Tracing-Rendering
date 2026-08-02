#ifndef FLAT_INTERSECTOR_CORE_H
#define FLAT_INTERSECTOR_CORE_H

#include "compiled_scene.h"
#include "rng.h"
#include "triangle_intersection.h"

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>

namespace packed_intersector {

constexpr std::uint32_t kTraversalStackSize = 64;

RT_HOST_DEVICE RT_FORCE_INLINE float minimum(float a, float b) {
    return a < b ? a : b;
}

RT_HOST_DEVICE RT_FORCE_INLINE float maximum(float a, float b) {
    return a > b ? a : b;
}

RT_HOST_DEVICE RT_FORCE_INLINE float absolute(float value) {
    return value < 0.0f ? -value : value;
}

RT_HOST_DEVICE RT_FORCE_INLINE bool finite(float value) {
    return value == value && value <= FLT_MAX && value >= -FLT_MAX;
}

RT_HOST_DEVICE RT_FORCE_INLINE float multiply_add(float a, float b, float c) {
    return ::fmaf(a, b, c);
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 add(Float3 a, Float3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 subtract(Float3 a, Float3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 multiply(Float3 value, float scalar) {
    return {value.x * scalar, value.y * scalar, value.z * scalar};
}

RT_HOST_DEVICE RT_FORCE_INLINE float dot_product(Float3 a, Float3 b) {
    return multiply_add(a.x, b.x,
                        multiply_add(a.y, b.y, a.z * b.z));
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 cross_product(Float3 a, Float3 b) {
    return {multiply_add(a.y, b.z, -a.z * b.y),
            multiply_add(a.z, b.x, -a.x * b.z),
            multiply_add(a.x, b.y, -a.y * b.x)};
}

RT_HOST_DEVICE RT_FORCE_INLINE float length(Float3 value) {
    return ::sqrtf(dot_product(value, value));
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 normalized(Float3 value) {
    const float magnitude = length(value);
    if (!(magnitude > 0.0f) || !finite(magnitude)) {
        return {};
    }
    return multiply(value, 1.0f / magnitude);
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 ray_at(const PackedRay &ray, float t) {
    return add(ray.origin, multiply(ray.direction, t));
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 transform_point(const float matrix[12],
                                                       Float3 point) {
    return {multiply_add(
                matrix[0], point.x,
                multiply_add(matrix[1], point.y,
                             multiply_add(matrix[2], point.z, matrix[3]))),
            multiply_add(
                matrix[4], point.x,
                multiply_add(matrix[5], point.y,
                             multiply_add(matrix[6], point.z, matrix[7]))),
            multiply_add(
                matrix[8], point.x,
                multiply_add(matrix[9], point.y,
                             multiply_add(matrix[10], point.z, matrix[11])))};
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 transform_vector(
    const float matrix[12], Float3 vector) {
    return {multiply_add(matrix[0], vector.x,
                         multiply_add(matrix[1], vector.y,
                                      matrix[2] * vector.z)),
            multiply_add(matrix[4], vector.x,
                         multiply_add(matrix[5], vector.y,
                                      matrix[6] * vector.z)),
            multiply_add(matrix[8], vector.x,
                         multiply_add(matrix[9], vector.y,
                                      matrix[10] * vector.z))};
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 transform_normal(
    const PackedTransform &transform, Float3 normal) {
    const float *matrix = transform.world_to_object;
    return {multiply_add(matrix[0], normal.x,
                         multiply_add(matrix[4], normal.y,
                                      matrix[8] * normal.z)),
            multiply_add(matrix[1], normal.x,
                         multiply_add(matrix[5], normal.y,
                                      matrix[9] * normal.z)),
            multiply_add(matrix[2], normal.x,
                         multiply_add(matrix[6], normal.y,
                                      matrix[10] * normal.z))};
}

RT_HOST_DEVICE RT_FORCE_INLINE float linear_determinant(
    const PackedTransform &transform) {
    const float *m = transform.object_to_world;
    return m[0] * (m[5] * m[10] - m[6] * m[9]) -
           m[1] * (m[4] * m[10] - m[6] * m[8]) +
           m[2] * (m[4] * m[9] - m[5] * m[8]);
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedRay transform_ray_to_object(
    const PackedRay &ray, const PackedTransform &transform) {
    PackedRay result = ray;
    result.origin = transform_point(transform.world_to_object, ray.origin);
    result.direction =
        transform_vector(transform.world_to_object, ray.direction);
    return result;
}

RT_HOST_DEVICE RT_FORCE_INLINE bool intersect_bounds(
    const PackedBVHNode &node, const PackedRay &ray, float t_min,
    float t_max) {
    const float origins[3]{ray.origin.x, ray.origin.y, ray.origin.z};
    const float directions[3]{ray.direction.x, ray.direction.y,
                              ray.direction.z};
    const float lower[3]{node.bounds_min.x, node.bounds_min.y,
                         node.bounds_min.z};
    const float upper[3]{node.bounds_max.x, node.bounds_max.y,
                         node.bounds_max.z};
    for (int axis = 0; axis < 3; ++axis) {
        if (absolute(directions[axis]) <= 1.17549435e-38f) {
            if (origins[axis] < lower[axis] || origins[axis] > upper[axis]) {
                return false;
            }
            continue;
        }
        const float inverse = 1.0f / directions[axis];
        float near_value = (lower[axis] - origins[axis]) * inverse;
        float far_value = (upper[axis] - origins[axis]) * inverse;
        if (near_value > far_value) {
            const float temporary = near_value;
            near_value = far_value;
            far_value = temporary;
        }
        t_min = maximum(t_min, near_value);
        t_max = minimum(t_max, far_value);
        if (t_max < t_min) {
            return false;
        }
    }
    return true;
}

RT_HOST_DEVICE RT_FORCE_INLINE bool intersect_bounds(
    Float3 lower, Float3 upper, const PackedRay &ray, float t_min,
    float t_max) {
    PackedBVHNode bounds{};
    bounds.bounds_min = lower;
    bounds.bounds_max = upper;
    return intersect_bounds(bounds, ray, t_min, t_max);
}

RT_HOST_DEVICE RT_FORCE_INLINE bool intersect_triangle(
    const CompiledSceneView &scene, const PackedMesh &mesh,
    std::uint32_t triangle_index, const PackedRay &object_ray, float t_min,
    float t_max, PackedHit &candidate) {
    const PackedTriangle &triangle = scene.triangles[triangle_index];
    const Float4 &packed_v0 =
        scene.positions[mesh.vertices.offset + triangle.vertex0];
    const Float4 &packed_v1 =
        scene.positions[mesh.vertices.offset + triangle.vertex1];
    const Float4 &packed_v2 =
        scene.positions[mesh.vertices.offset + triangle.vertex2];
    const Float3 v0{packed_v0.x, packed_v0.y, packed_v0.z};
    const Float3 v1{packed_v1.x, packed_v1.y, packed_v1.z};
    const Float3 v2{packed_v2.x, packed_v2.y, packed_v2.z};
    using namespace triangle_intersection;
    TriangleKernelHit<float> intersection;
    if (!intersect_triangle_kernel(
            {object_ray.origin.x, object_ray.origin.y, object_ray.origin.z},
            {object_ray.direction.x, object_ray.direction.y,
             object_ray.direction.z},
            {v0.x, v0.y, v0.z}, {v1.x, v1.y, v1.z}, {v2.x, v2.y, v2.z},
            t_min, t_max, intersection)) {
        return false;
    }
    candidate.t = intersection.t;
    candidate.barycentric_u = intersection.barycentric_u;
    candidate.barycentric_v = intersection.barycentric_v;
    candidate.primitive_id = triangle_index;
    candidate.flags = PACKED_HIT_TRIANGLE;
    return true;
}

RT_HOST_DEVICE RT_FORCE_INLINE bool intersect_mesh(
    const CompiledSceneView &scene, const PackedMesh &mesh,
    const PackedRay &object_ray, float t_min, float t_max, PackedHit &hit,
    PackedTraversalStatus &status) {
    if (mesh.bvh_nodes.count == 0) {
        return false;
    }
    bool found = false;
    std::uint32_t stack[kTraversalStackSize]{};
    std::uint32_t stack_size = 0;
    stack[stack_size++] = mesh.bvh_nodes.offset;
    while (stack_size != 0) {
        const std::uint32_t node_index = stack[--stack_size];
        const PackedBVHNode &node = scene.bvh_nodes[node_index];
        if (!intersect_bounds(node, object_ray, t_min, t_max)) {
            continue;
        }
        if (node.is_leaf()) {
            for (std::uint32_t local = 0; local < node.primitive_count();
                 ++local) {
                PackedHit candidate{};
                if (intersect_triangle(scene, mesh, node.first + local,
                                       object_ray, t_min, t_max,
                                       candidate)) {
                    t_max = candidate.t;
                    hit = candidate;
                    found = true;
                }
            }
            continue;
        }
        if (stack_size + 2 > kTraversalStackSize) {
            status = PackedTraversalStatus::StackOverflow;
            return false;
        }
        stack[stack_size++] = node.first;
        stack[stack_size++] = node_index + 1;
    }
    return found;
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 moving_sphere_center(
    const PackedMovingSphere &sphere, float time) {
    const float duration = sphere.time1 - sphere.time0;
    if (absolute(duration) <= 1e-15f) {
        return sphere.center0;
    }
    const float offset = (time - sphere.time0) / duration;
    return add(sphere.center0,
               multiply(subtract(sphere.center1, sphere.center0), offset));
}

RT_HOST_DEVICE RT_FORCE_INLINE bool intersect_sphere(
    Float3 center, float radius, const PackedRay &ray, float t_min,
    float t_max, PackedHit &hit) {
    const Float3 oc = subtract(ray.origin, center);
    const float a = dot_product(ray.direction, ray.direction);
    const float half_b = dot_product(oc, ray.direction);
    const float c = multiply_add(-radius, radius, dot_product(oc, oc));
    const float discriminant = multiply_add(half_b, half_b, -a * c);
    if (discriminant < 0.0f || !(a > 0.0f)) {
        return false;
    }
    const float root_term = ::sqrtf(discriminant);
    float root = (-half_b - root_term) / a;
    if (root < t_min || root > t_max) {
        root = (-half_b + root_term) / a;
        if (root < t_min || root > t_max) {
            return false;
        }
    }
    hit.t = root;
    hit.flags = PACKED_HIT_SPHERE;
    return true;
}

RT_HOST_DEVICE RT_FORCE_INLINE std::uint32_t resolve_material(
    const CompiledSceneView &scene, const PackedInstance &instance,
    std::uint32_t slot) {
    if (slot >= instance.material_bindings.count) {
        return kInvalidPackedIndex;
    }
    return scene.material_bindings[instance.material_bindings.offset + slot];
}

RT_HOST_DEVICE RT_FORCE_INLINE bool intersect_instance_surface(
    const CompiledSceneView &scene, std::uint32_t instance_id,
    const PackedRay &ray, float t_max, PackedHit &hit,
    PackedTraversalStatus &status) {
    const PackedInstance &instance = scene.instances[instance_id];
    if (!intersect_bounds(instance.bounds_min, instance.bounds_max, ray,
                          ray.t_min, t_max)) {
        return false;
    }
    if (instance.geometry_type == PackedGeometryType::Medium) {
        return false;
    }
    const PackedTransform &transform = scene.transforms[instance.transform_id];
    const PackedRay object_ray = transform_ray_to_object(ray, transform);
    PackedHit candidate{};
    bool found = false;
    switch (instance.geometry_type) {
    case PackedGeometryType::Mesh:
        found = intersect_mesh(scene, scene.meshes[instance.geometry_index],
                               object_ray, ray.t_min, t_max, candidate,
                               status);
        break;
    case PackedGeometryType::Sphere: {
        const PackedSphere &sphere = scene.spheres[instance.geometry_index];
        found = intersect_sphere(sphere.center, sphere.radius, object_ray,
                                 ray.t_min, t_max, candidate);
        break;
    }
    case PackedGeometryType::MovingSphere: {
        const PackedMovingSphere &sphere =
            scene.moving_spheres[instance.geometry_index];
        found = intersect_sphere(moving_sphere_center(sphere, ray.time),
                                 sphere.radius, object_ray, ray.t_min, t_max,
                                 candidate);
        break;
    }
    case PackedGeometryType::Medium:
        break;
    }
    if (!found || status == PackedTraversalStatus::StackOverflow) {
        return false;
    }
    candidate.instance_id = instance_id;
    if ((candidate.flags & PACKED_HIT_TRIANGLE) == 0) {
        candidate.primitive_id = instance.geometry_index;
    }
    hit = candidate;
    return true;
}

RT_HOST_DEVICE RT_FORCE_INLINE bool intersect_aggregate_surface(
    const CompiledSceneView &scene, std::uint32_t aggregate_id,
    const PackedRay &ray, PackedHit &hit, PackedTraversalStatus &status) {
    if (aggregate_id >= scene.aggregates.count) {
        return false;
    }
    const PackedAggregate &aggregate = scene.aggregates[aggregate_id];
    if (aggregate.bvh_nodes.count == 0) {
        return false;
    }
    bool found = false;
    float closest = ray.t_max;
    std::uint32_t stack[kTraversalStackSize]{};
    std::uint32_t stack_size = 0;
    stack[stack_size++] = aggregate.bvh_nodes.offset;
    while (stack_size != 0) {
        const std::uint32_t node_index = stack[--stack_size];
        const PackedBVHNode &node = scene.bvh_nodes[node_index];
        if (!intersect_bounds(node, ray, ray.t_min, closest)) {
            continue;
        }
        if (node.is_leaf()) {
            for (std::uint32_t local = 0; local < node.primitive_count();
                 ++local) {
                const std::uint32_t instance_id =
                    scene.aggregate_instance_indices[node.first + local];
                PackedHit candidate{};
                if (intersect_instance_surface(scene, instance_id, ray,
                                               closest, candidate, status)) {
                    closest = candidate.t;
                    hit = candidate;
                    found = true;
                }
                if (status == PackedTraversalStatus::StackOverflow) {
                    return false;
                }
            }
            continue;
        }
        if (stack_size + 2 > kTraversalStackSize) {
            status = PackedTraversalStatus::StackOverflow;
            return false;
        }
        stack[stack_size++] = node.first;
        stack[stack_size++] = node_index + 1;
    }
    return found;
}

RT_HOST_DEVICE RT_FORCE_INLINE bool intersect_medium(
    const CompiledSceneView &scene, const PackedMedium &medium,
    const PackedRay &ray, float t_max, PackedHit &hit, RNG *rng,
    PackedTraversalStatus &status) {
    if (rng == nullptr) {
        return false;
    }
    PackedRay boundary_ray = ray;
    boundary_ray.t_min = -std::numeric_limits<float>::infinity();
    boundary_ray.t_max = std::numeric_limits<float>::infinity();
    PackedHit entry{};
    if (!intersect_aggregate_surface(scene, medium.boundary_aggregate,
                                     boundary_ray, entry, status)) {
        return false;
    }
    boundary_ray.t_min = entry.t + 0.0001f;
    PackedHit exit{};
    if (!intersect_aggregate_surface(scene, medium.boundary_aggregate,
                                     boundary_ray, exit, status)) {
        return false;
    }

    float entry_t = maximum(entry.t, ray.t_min);
    const float exit_t = minimum(exit.t, t_max);
    if (entry_t >= exit_t) {
        return false;
    }
    entry_t = maximum(entry_t, 0.0f);
    const float ray_length = length(ray.direction);
    if (!(ray_length > 0.0f)) {
        return false;
    }
    const float hit_distance =
        medium.neg_inv_density * ::logf(static_cast<float>(rng->next()));
    if (hit_distance > (exit_t - entry_t) * ray_length) {
        return false;
    }
    hit.t = entry_t + hit_distance / ray_length;
    hit.flags = PACKED_HIT_MEDIUM | PACKED_HIT_FRONT_FACE;
    return true;
}

RT_HOST_DEVICE RT_FORCE_INLINE bool intersect_instance(
    const CompiledSceneView &scene, std::uint32_t instance_id,
    const PackedRay &ray, float t_max, PackedHit &hit, RNG *rng,
    PackedTraversalStatus &status) {
    const PackedInstance &instance = scene.instances[instance_id];
    if (instance.geometry_type != PackedGeometryType::Medium) {
        return intersect_instance_surface(scene, instance_id, ray, t_max,
                                          hit, status);
    }
    if (!intersect_bounds(instance.bounds_min, instance.bounds_max, ray,
                          ray.t_min, t_max)) {
        return false;
    }
    PackedHit candidate{};
    if (!intersect_medium(scene, scene.media[instance.geometry_index], ray,
                          t_max, candidate, rng, status)) {
        return false;
    }
    candidate.instance_id = instance_id;
    candidate.primitive_id = instance.geometry_index;
    hit = candidate;
    return true;
}

RT_HOST_DEVICE RT_FORCE_INLINE bool intersect_aggregate(
    const CompiledSceneView &scene, std::uint32_t aggregate_id,
    const PackedRay &ray, PackedHit &hit, RNG *rng,
    PackedTraversalStatus &status) {
    if (aggregate_id >= scene.aggregates.count) {
        return false;
    }
    const PackedAggregate &aggregate = scene.aggregates[aggregate_id];
    if (aggregate.bvh_nodes.count == 0) {
        return false;
    }
    bool found = false;
    float closest = ray.t_max;
    std::uint32_t stack[kTraversalStackSize]{};
    std::uint32_t stack_size = 0;
    stack[stack_size++] = aggregate.bvh_nodes.offset;
    while (stack_size != 0) {
        const std::uint32_t node_index = stack[--stack_size];
        const PackedBVHNode &node = scene.bvh_nodes[node_index];
        if (!intersect_bounds(node, ray, ray.t_min, closest)) {
            continue;
        }
        if (node.is_leaf()) {
            for (std::uint32_t local = 0; local < node.primitive_count();
                 ++local) {
                const std::uint32_t instance_id =
                    scene.aggregate_instance_indices[node.first + local];
                PackedHit candidate{};
                if (intersect_instance(scene, instance_id, ray, closest,
                                       candidate, rng, status)) {
                    closest = candidate.t;
                    hit = candidate;
                    found = true;
                }
                if (status == PackedTraversalStatus::StackOverflow) {
                    return false;
                }
            }
            continue;
        }
        if (stack_size + 2 > kTraversalStackSize) {
            status = PackedTraversalStatus::StackOverflow;
            return false;
        }
        stack[stack_size++] = node.first;
        stack[stack_size++] = node_index + 1;
    }
    return found;
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedTraversalStatus
intersect_compiled_scene_core(const CompiledSceneView &scene,
                              const PackedRay &ray, PackedHit &hit,
                              RNG *rng) {
    if (scene.aggregates.count == 0 || !finite(ray.time) ||
        ray.t_max < ray.t_min) {
        return PackedTraversalStatus::InvalidInput;
    }
    PackedTraversalStatus status = PackedTraversalStatus::Miss;
    const bool found = intersect_aggregate(scene, 0, ray, hit, rng, status);
    if (status == PackedTraversalStatus::StackOverflow) {
        return status;
    }
    return found ? PackedTraversalStatus::Hit : PackedTraversalStatus::Miss;
}

} // namespace packed_intersector

#endif

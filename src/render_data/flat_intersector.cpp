#include "flat_intersector.h"

#include "rtweekend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace {

constexpr std::size_t kTraversalStackSize = 64;
constexpr float kTriangleEpsilon = 1e-8f;

Float3 add(Float3 a, Float3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Float3 subtract(Float3 a, Float3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Float3 multiply(Float3 value, float scalar) {
    return {value.x * scalar, value.y * scalar, value.z * scalar};
}

float dot_product(Float3 a, Float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Float3 cross_product(Float3 a, Float3 b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

float length(Float3 value) {
    return std::sqrt(dot_product(value, value));
}

Float3 normalized(Float3 value) {
    const float magnitude = length(value);
    if (!(magnitude > 0.0f) || !std::isfinite(magnitude)) {
        return {};
    }
    return multiply(value, 1.0f / magnitude);
}

Float3 ray_at(const PackedRay &ray, float t) {
    return add(ray.origin, multiply(ray.direction, t));
}

Float3 transform_point(const float matrix[12], Float3 point) {
    return {matrix[0] * point.x + matrix[1] * point.y +
                matrix[2] * point.z + matrix[3],
            matrix[4] * point.x + matrix[5] * point.y +
                matrix[6] * point.z + matrix[7],
            matrix[8] * point.x + matrix[9] * point.y +
                matrix[10] * point.z + matrix[11]};
}

Float3 transform_vector(const float matrix[12], Float3 vector) {
    return {matrix[0] * vector.x + matrix[1] * vector.y +
                matrix[2] * vector.z,
            matrix[4] * vector.x + matrix[5] * vector.y +
                matrix[6] * vector.z,
            matrix[8] * vector.x + matrix[9] * vector.y +
                matrix[10] * vector.z};
}

Float3 transform_normal(const PackedTransform &transform, Float3 normal) {
    const float *matrix = transform.world_to_object;
    return {matrix[0] * normal.x + matrix[4] * normal.y +
                matrix[8] * normal.z,
            matrix[1] * normal.x + matrix[5] * normal.y +
                matrix[9] * normal.z,
            matrix[2] * normal.x + matrix[6] * normal.y +
                matrix[10] * normal.z};
}

float linear_determinant(const PackedTransform &transform) {
    const float *m = transform.object_to_world;
    return m[0] * (m[5] * m[10] - m[6] * m[9]) -
           m[1] * (m[4] * m[10] - m[6] * m[8]) +
           m[2] * (m[4] * m[9] - m[5] * m[8]);
}

PackedRay transform_ray_to_object(const PackedRay &ray,
                                  const PackedTransform &transform) {
    PackedRay result = ray;
    result.origin = transform_point(transform.world_to_object, ray.origin);
    result.direction =
        transform_vector(transform.world_to_object, ray.direction);
    return result;
}

bool intersect_bounds(const PackedBVHNode &node, const PackedRay &ray,
                      float t_min, float t_max) {
    const float origins[3]{ray.origin.x, ray.origin.y, ray.origin.z};
    const float directions[3]{ray.direction.x, ray.direction.y,
                              ray.direction.z};
    const float minimum[3]{node.bounds_min.x, node.bounds_min.y,
                           node.bounds_min.z};
    const float maximum[3]{node.bounds_max.x, node.bounds_max.y,
                           node.bounds_max.z};
    for (int axis = 0; axis < 3; ++axis) {
        if (std::abs(directions[axis]) <=
            std::numeric_limits<float>::min()) {
            if (origins[axis] < minimum[axis] ||
                origins[axis] > maximum[axis]) {
                return false;
            }
            continue;
        }
        const float inverse = 1.0f / directions[axis];
        float near = (minimum[axis] - origins[axis]) * inverse;
        float far = (maximum[axis] - origins[axis]) * inverse;
        if (near > far) {
            std::swap(near, far);
        }
        t_min = std::max(t_min, near);
        t_max = std::min(t_max, far);
        if (t_max < t_min) {
            return false;
        }
    }
    return true;
}

bool intersect_bounds(Float3 minimum, Float3 maximum,
                      const PackedRay &ray, float t_min, float t_max) {
    PackedBVHNode bounds;
    bounds.bounds_min = minimum;
    bounds.bounds_max = maximum;
    return intersect_bounds(bounds, ray, t_min, t_max);
}

bool intersect_triangle(const CompiledSceneView &scene,
                        const PackedMesh &mesh,
                        std::uint32_t triangle_index,
                        const PackedRay &object_ray, float t_min,
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
    const Float3 edge1 = subtract(v1, v0);
    const Float3 edge2 = subtract(v2, v0);
    const Float3 pvec = cross_product(object_ray.direction, edge2);
    const float determinant = dot_product(edge1, pvec);
    if (std::abs(determinant) < kTriangleEpsilon) {
        return false;
    }
    const float inverse = 1.0f / determinant;
    const Float3 tvec = subtract(object_ray.origin, v0);
    const float u = dot_product(tvec, pvec) * inverse;
    if (u < 0.0f || u > 1.0f) {
        return false;
    }
    const Float3 qvec = cross_product(tvec, edge1);
    const float v = dot_product(object_ray.direction, qvec) * inverse;
    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }
    const float t = dot_product(edge2, qvec) * inverse;
    if (t < t_min || t > t_max || !std::isfinite(t)) {
        return false;
    }
    candidate.t = t;
    candidate.barycentric_u = u;
    candidate.barycentric_v = v;
    candidate.primitive_id = triangle_index;
    candidate.flags = PACKED_HIT_TRIANGLE;
    return true;
}

bool intersect_mesh(const CompiledSceneView &scene, const PackedMesh &mesh,
                    const PackedRay &object_ray, float t_min, float t_max,
                    PackedHit &hit) {
    if (mesh.bvh_nodes.count == 0) {
        return false;
    }
    bool found = false;
    std::array<std::uint32_t, kTraversalStackSize> stack{};
    std::size_t stack_size = 0;
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
                PackedHit candidate;
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
        if (stack_size + 2 > stack.size()) {
            return false;
        }
        stack[stack_size++] = node.first;
        stack[stack_size++] = node_index + 1;
    }
    return found;
}

Float3 moving_sphere_center(const PackedMovingSphere &sphere, float time) {
    const float duration = sphere.time1 - sphere.time0;
    if (std::abs(duration) <= 1e-15f) {
        return sphere.center0;
    }
    const float t = (time - sphere.time0) / duration;
    return add(sphere.center0,
               multiply(subtract(sphere.center1, sphere.center0), t));
}

bool intersect_sphere(Float3 center, float radius, const PackedRay &ray,
                      float t_min, float t_max, PackedHit &hit) {
    const Float3 oc = subtract(ray.origin, center);
    const float a = dot_product(ray.direction, ray.direction);
    const float half_b = dot_product(oc, ray.direction);
    const float c = dot_product(oc, oc) - radius * radius;
    const float discriminant = half_b * half_b - a * c;
    if (discriminant < 0.0f || !(a > 0.0f)) {
        return false;
    }
    const float root_term = std::sqrt(discriminant);
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

std::uint32_t resolve_material(const CompiledSceneView &scene,
                               const PackedInstance &instance,
                               std::uint32_t slot) {
    if (slot >= instance.material_bindings.count) {
        return kInvalidPackedIndex;
    }
    return scene.material_bindings[instance.material_bindings.offset + slot];
}

bool intersect_aggregate(const CompiledSceneView &scene,
                         std::uint32_t aggregate_id, const PackedRay &ray,
                         PackedHit &hit, RNG *rng, bool allow_media);

bool intersect_medium(const CompiledSceneView &scene,
                      const PackedMedium &medium, const PackedRay &ray,
                      float t_max, PackedHit &hit, RNG *rng) {
    if (rng == nullptr) {
        return false;
    }
    PackedRay boundary_ray = ray;
    boundary_ray.t_min = -std::numeric_limits<float>::infinity();
    boundary_ray.t_max = std::numeric_limits<float>::infinity();
    PackedHit entry;
    if (!intersect_aggregate(scene, medium.boundary_aggregate, boundary_ray,
                             entry, nullptr, false)) {
        return false;
    }
    boundary_ray.t_min = entry.t + 0.0001f;
    PackedHit exit;
    if (!intersect_aggregate(scene, medium.boundary_aggregate, boundary_ray,
                             exit, nullptr, false)) {
        return false;
    }

    float entry_t = std::max(entry.t, ray.t_min);
    const float exit_t = std::min(exit.t, t_max);
    if (entry_t >= exit_t) {
        return false;
    }
    entry_t = std::max(entry_t, 0.0f);
    const float ray_length = length(ray.direction);
    if (!(ray_length > 0.0f)) {
        return false;
    }
    const float hit_distance = medium.neg_inv_density *
                               std::log(static_cast<float>(rng->next()));
    if (hit_distance > (exit_t - entry_t) * ray_length) {
        return false;
    }
    hit.t = entry_t + hit_distance / ray_length;
    hit.flags = PACKED_HIT_MEDIUM | PACKED_HIT_FRONT_FACE;
    return true;
}

bool intersect_instance(const CompiledSceneView &scene,
                        std::uint32_t instance_id, const PackedRay &ray,
                        float t_max, PackedHit &hit, RNG *rng,
                        bool allow_media) {
    const PackedInstance &instance = scene.instances[instance_id];
    if (!intersect_bounds(instance.bounds_min, instance.bounds_max, ray,
                          ray.t_min, t_max)) {
        return false;
    }
    const PackedTransform &transform = scene.transforms[instance.transform_id];
    const PackedRay object_ray = transform_ray_to_object(ray, transform);
    PackedHit candidate;
    bool found = false;
    switch (instance.geometry_type) {
    case PackedGeometryType::Mesh: {
        const PackedMesh &mesh = scene.meshes[instance.geometry_index];
        found = intersect_mesh(scene, mesh, object_ray, ray.t_min, t_max,
                               candidate);
        break;
    }
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
        if (allow_media) {
            found = intersect_medium(scene, scene.media[instance.geometry_index],
                                     ray, t_max, candidate, rng);
        }
        break;
    }
    if (!found) {
        return false;
    }
    candidate.instance_id = instance_id;
    if ((candidate.flags & PACKED_HIT_TRIANGLE) == 0) {
        candidate.primitive_id = instance.geometry_index;
    }
    hit = candidate;
    return true;
}

bool intersect_aggregate(const CompiledSceneView &scene,
                         std::uint32_t aggregate_id, const PackedRay &ray,
                         PackedHit &hit, RNG *rng, bool allow_media) {
    if (aggregate_id >= scene.aggregates.count) {
        return false;
    }
    const PackedAggregate &aggregate = scene.aggregates[aggregate_id];
    if (aggregate.bvh_nodes.count == 0) {
        return false;
    }
    bool found = false;
    float closest = ray.t_max;
    std::array<std::uint32_t, kTraversalStackSize> stack{};
    std::size_t stack_size = 0;
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
                PackedHit candidate;
                if (intersect_instance(scene, instance_id, ray, closest,
                                       candidate, rng, allow_media)) {
                    closest = candidate.t;
                    hit = candidate;
                    found = true;
                }
            }
            continue;
        }
        if (stack_size + 2 > stack.size()) {
            return false;
        }
        stack[stack_size++] = node.first;
        stack[stack_size++] = node_index + 1;
    }
    return found;
}

Float3 packed_position(const CompiledSceneView &scene, const PackedMesh &mesh,
                       std::uint32_t index) {
    const Float4 &value = scene.positions[mesh.vertices.offset + index];
    return {value.x, value.y, value.z};
}

Float3 packed_normal(const CompiledSceneView &scene, const PackedMesh &mesh,
                     std::uint32_t index) {
    const Float4 &value = scene.normals[mesh.vertices.offset + index];
    return {value.x, value.y, value.z};
}

Float3 interpolate(Float3 a, Float3 b, Float3 c, float u, float v) {
    return add(multiply(a, 1.0f - u - v),
               add(multiply(b, u), multiply(c, v)));
}

Float2 interpolate(Float2 a, Float2 b, Float2 c, float u, float v) {
    const float w = 1.0f - u - v;
    return {w * a.x + u * b.x + v * c.x,
            w * a.y + u * b.y + v * c.y};
}

Float4 interpolate(Float4 a, Float4 b, Float4 c, float u, float v) {
    const float w = 1.0f - u - v;
    return {w * a.x + u * b.x + v * c.x,
            w * a.y + u * b.y + v * c.y,
            w * a.z + u * b.z + v * c.z,
            w * a.w + u * b.w + v * c.w};
}

bool reconstruct_triangle(const CompiledSceneView &scene,
                          const PackedRay &ray, const PackedHit &hit,
                          const PackedInstance &instance,
                          const PackedTransform &transform,
                          PackedSurfaceInteraction &surface) {
    if (instance.geometry_index >= scene.meshes.count ||
        hit.primitive_id >= scene.triangles.count) {
        return false;
    }
    const PackedMesh &mesh = scene.meshes[instance.geometry_index];
    const PackedTriangle &triangle = scene.triangles[hit.primitive_id];
    const Float3 object_v0 = packed_position(scene, mesh, triangle.vertex0);
    const Float3 object_v1 = packed_position(scene, mesh, triangle.vertex1);
    const Float3 object_v2 = packed_position(scene, mesh, triangle.vertex2);
    const Float3 object_edge1 = subtract(object_v1, object_v0);
    const Float3 object_edge2 = subtract(object_v2, object_v0);
    const Float3 world_edge1 =
        transform_vector(transform.object_to_world, object_edge1);
    const Float3 world_edge2 =
        transform_vector(transform.object_to_world, object_edge2);
    Float3 outward = normalized(cross_product(world_edge1, world_edge2));
    Float3 object_shading =
        normalized(cross_product(object_edge1, object_edge2));
    Float3 shading = outward;
    if ((triangle.flags & PACKED_TRIANGLE_HAS_NORMALS) != 0) {
        object_shading = normalized(interpolate(
            packed_normal(scene, mesh, triangle.vertex0),
            packed_normal(scene, mesh, triangle.vertex1),
            packed_normal(scene, mesh, triangle.vertex2),
            hit.barycentric_u, hit.barycentric_v));
        shading = normalized(transform_normal(transform, object_shading));
        if (dot_product(shading, outward) < 0.0f) {
            shading = multiply(shading, -1.0f);
        }
    }
    if ((instance.flags & PACKED_INSTANCE_FLIP_FACE) != 0) {
        outward = multiply(outward, -1.0f);
        shading = multiply(shading, -1.0f);
    }

    Float3 object_dpdu = object_edge1;
    Float3 object_dpdv = object_edge2;
    const Float2 uv0 = scene.uv0[mesh.vertices.offset + triangle.vertex0];
    const Float2 uv1 = scene.uv0[mesh.vertices.offset + triangle.vertex1];
    const Float2 uv2 = scene.uv0[mesh.vertices.offset + triangle.vertex2];
    if ((triangle.flags & PACKED_TRIANGLE_HAS_TANGENT) != 0) {
        const Float4 t0 =
            scene.tangents[mesh.vertices.offset + triangle.vertex0];
        const Float4 t1 =
            scene.tangents[mesh.vertices.offset + triangle.vertex1];
        const Float4 t2 =
            scene.tangents[mesh.vertices.offset + triangle.vertex2];
        Float3 tangent = interpolate(Float3{t0.x, t0.y, t0.z},
                                     Float3{t1.x, t1.y, t1.z},
                                     Float3{t2.x, t2.y, t2.z},
                                     hit.barycentric_u, hit.barycentric_v);
        tangent = subtract(tangent,
                           multiply(object_shading,
                                    dot_product(tangent, object_shading)));
        tangent = normalized(tangent);
        if (length(tangent) > 0.0f) {
            const float w = 1.0f - hit.barycentric_u - hit.barycentric_v;
            const float sign =
                w * t0.w + hit.barycentric_u * t1.w +
                            hit.barycentric_v * t2.w <
                        0.0f
                    ? -1.0f
                    : 1.0f;
            object_dpdu = tangent;
            object_dpdv =
                multiply(cross_product(object_shading, tangent), sign);
        }
    } else if ((triangle.flags & PACKED_TRIANGLE_HAS_UV) != 0) {
        const Float2 duv1{uv1.x - uv0.x, uv1.y - uv0.y};
        const Float2 duv2{uv2.x - uv0.x, uv2.y - uv0.y};
        const float determinant = duv1.x * duv2.y - duv1.y * duv2.x;
        if (std::abs(determinant) > 1e-10f) {
            const float inverse = 1.0f / determinant;
            object_dpdu = multiply(
                subtract(multiply(object_edge1, duv2.y),
                         multiply(object_edge2, duv1.y)),
                inverse);
            object_dpdv = multiply(
                add(multiply(object_edge1, -duv2.x),
                    multiply(object_edge2, duv1.x)),
                inverse);
        }
    }

    const bool front_face = dot_product(ray.direction, outward) < 0.0f;
    surface.position = ray_at(ray, hit.t);
    surface.geometric_normal =
        front_face ? outward : multiply(outward, -1.0f);
    surface.shading_normal =
        front_face ? shading : multiply(shading, -1.0f);
    surface.dpdu =
        transform_vector(transform.object_to_world, object_dpdu);
    surface.dpdv =
        transform_vector(transform.object_to_world, object_dpdv);
    surface.uv = (triangle.flags & PACKED_TRIANGLE_HAS_UV) != 0
                     ? interpolate(uv0, uv1, uv2, hit.barycentric_u,
                                   hit.barycentric_v)
                     : Float2{hit.barycentric_u, hit.barycentric_v};
    if ((triangle.flags & PACKED_TRIANGLE_HAS_COLOR) != 0) {
        surface.vertex_color = interpolate(
            scene.vertex_colors[mesh.vertices.offset + triangle.vertex0],
            scene.vertex_colors[mesh.vertices.offset + triangle.vertex1],
            scene.vertex_colors[mesh.vertices.offset + triangle.vertex2],
            hit.barycentric_u, hit.barycentric_v);
        surface.vertex_alpha = surface.vertex_color.w;
    }
    surface.flags = PACKED_HIT_TRIANGLE |
                    (front_face ? PACKED_HIT_FRONT_FACE : 0u);
    surface.material_id =
        resolve_material(scene, instance, triangle.material_slot);
    surface.primitive_id = triangle.primitive_id;
    return true;
}

void sphere_uv(Float3 normal, Float2 &uv) {
    constexpr float pi_f = 3.14159265358979323846f;
    const float theta = std::acos(std::clamp(-normal.y, -1.0f, 1.0f));
    const float phi = std::atan2(-normal.z, normal.x) + pi_f;
    uv = {phi / (2.0f * pi_f), theta / pi_f};
}

bool reconstruct_sphere(const CompiledSceneView &scene,
                        const PackedRay &ray, const PackedHit &hit,
                        const PackedInstance &instance,
                        const PackedTransform &transform,
                        PackedSurfaceInteraction &surface) {
    Float3 center;
    float radius = 0.0f;
    std::uint32_t sphere_flags = 0;
    if (instance.geometry_type == PackedGeometryType::Sphere) {
        const PackedSphere &sphere = scene.spheres[instance.geometry_index];
        center = sphere.center;
        radius = sphere.radius;
        sphere_flags = sphere.flags;
    } else {
        const PackedMovingSphere &sphere =
            scene.moving_spheres[instance.geometry_index];
        center = moving_sphere_center(sphere, ray.time);
        radius = sphere.radius;
        sphere_flags = sphere.flags;
    }
    const PackedRay object_ray = transform_ray_to_object(ray, transform);
    const Float3 object_position = ray_at(object_ray, hit.t);
    Float3 object_outward =
        multiply(subtract(object_position, center), 1.0f / radius);
    const bool negative_radius =
        (sphere_flags & PACKED_SPHERE_FLIP_ORIENTATION) != 0;
    if (negative_radius) {
        object_outward = multiply(object_outward, -1.0f);
    }
    sphere_uv(object_outward, surface.uv);
    Float3 outward = normalized(transform_normal(transform, object_outward));
    if (linear_determinant(transform) < 0.0f) {
        outward = multiply(outward, -1.0f);
    }
    if ((instance.flags & PACKED_INSTANCE_FLIP_FACE) != 0) {
        outward = multiply(outward, -1.0f);
    }
    const bool front_face = dot_product(ray.direction, outward) < 0.0f;
    surface.position = ray_at(ray, hit.t);
    surface.geometric_normal =
        front_face ? outward : multiply(outward, -1.0f);
    surface.shading_normal = surface.geometric_normal;

    constexpr float pi_f = 3.14159265358979323846f;
    const float signed_radius = negative_radius ? -radius : radius;
    const float theta = surface.uv.y * pi_f;
    const float alpha = surface.uv.x * 2.0f * pi_f - pi_f;
    const float sin_theta = std::sin(theta);
    const float cos_theta = std::cos(theta);
    const float sin_alpha = std::sin(alpha);
    const float cos_alpha = std::cos(alpha);
    const Float3 object_dpdu =
        multiply({-sin_alpha * sin_theta, 0.0f,
                  -cos_alpha * sin_theta},
                 2.0f * pi_f * signed_radius);
    const Float3 object_dpdv =
        multiply({cos_alpha * cos_theta, sin_theta,
                  -sin_alpha * cos_theta},
                 pi_f * signed_radius);
    surface.dpdu =
        transform_vector(transform.object_to_world, object_dpdu);
    surface.dpdv =
        transform_vector(transform.object_to_world, object_dpdv);
    surface.flags = PACKED_HIT_SPHERE |
                    (front_face ? PACKED_HIT_FRONT_FACE : 0u);
    surface.material_id = resolve_material(scene, instance, 0);
    surface.primitive_id = instance.source_object_id;
    return true;
}

} // namespace

bool intersect_compiled_scene(const CompiledSceneView &scene,
                              const PackedRay &ray, PackedHit &hit,
                              RNG *rng) {
    if (scene.aggregates.count == 0 || !std::isfinite(ray.time) ||
        ray.t_max < ray.t_min) {
        return false;
    }
    return intersect_aggregate(scene, 0, ray, hit, rng, true);
}

bool reconstruct_compiled_hit(const CompiledSceneView &scene,
                              const PackedRay &ray, const PackedHit &hit,
                              PackedSurfaceInteraction &surface) {
    if (hit.instance_id >= scene.instances.count || !std::isfinite(hit.t)) {
        return false;
    }
    const PackedInstance &instance = scene.instances[hit.instance_id];
    if (instance.transform_id >= scene.transforms.count) {
        return false;
    }
    surface = {};
    surface.instance_id = hit.instance_id;
    const PackedTransform &transform = scene.transforms[instance.transform_id];
    if ((hit.flags & PACKED_HIT_TRIANGLE) != 0) {
        return reconstruct_triangle(scene, ray, hit, instance, transform,
                                    surface);
    }
    if ((hit.flags & PACKED_HIT_SPHERE) != 0) {
        return reconstruct_sphere(scene, ray, hit, instance, transform,
                                  surface);
    }
    if ((hit.flags & PACKED_HIT_MEDIUM) != 0) {
        surface.position = ray_at(ray, hit.t);
        surface.geometric_normal = {1.0f, 0.0f, 0.0f};
        surface.shading_normal = surface.geometric_normal;
        surface.dpdu = {0.0f, 1.0f, 0.0f};
        surface.dpdv = {0.0f, 0.0f, 1.0f};
        surface.material_id =
            scene.media[instance.geometry_index].phase_material;
        surface.primitive_id = instance.source_object_id;
        surface.flags = PACKED_HIT_MEDIUM | PACKED_HIT_FRONT_FACE;
        return surface.material_id < scene.materials.count;
    }
    return false;
}

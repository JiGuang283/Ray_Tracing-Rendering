#ifndef SURFACE_RECONSTRUCTION_CORE_H
#define SURFACE_RECONSTRUCTION_CORE_H

#include "flat_intersector_core.h"

#include <cmath>
#include <cstdint>

namespace packed_reconstruction {

using namespace packed_intersector;

RT_HOST_DEVICE RT_FORCE_INLINE std::uint32_t resolve_emitter(
    const CompiledSceneView &scene, const PackedInstance &instance,
    std::uint32_t material_slot) {
    if (material_slot >= instance.material_bindings.count) {
        return kInvalidPackedIndex;
    }
    const std::uint32_t binding =
        instance.material_bindings.offset + material_slot;
    return binding < scene.emitter_bindings.count
               ? scene.emitter_bindings[binding]
               : kInvalidPackedIndex;
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 packed_position(
    const CompiledSceneView &scene, const PackedMesh &mesh,
    std::uint32_t index) {
    const Float4 &value = scene.positions[mesh.vertices.offset + index];
    return {value.x, value.y, value.z};
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 packed_normal(
    const CompiledSceneView &scene, const PackedMesh &mesh,
    std::uint32_t index) {
    const Float4 &value = scene.normals[mesh.vertices.offset + index];
    return {value.x, value.y, value.z};
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 interpolate(
    Float3 a, Float3 b, Float3 c, float u, float v) {
    return add(multiply(a, 1.0f - u - v),
               add(multiply(b, u), multiply(c, v)));
}

RT_HOST_DEVICE RT_FORCE_INLINE Float2 interpolate(
    Float2 a, Float2 b, Float2 c, float u, float v) {
    const float w = 1.0f - u - v;
    return {w * a.x + u * b.x + v * c.x,
            w * a.y + u * b.y + v * c.y};
}

RT_HOST_DEVICE RT_FORCE_INLINE Float4 interpolate(
    Float4 a, Float4 b, Float4 c, float u, float v) {
    const float w = 1.0f - u - v;
    return {w * a.x + u * b.x + v * c.x,
            w * a.y + u * b.y + v * c.y,
            w * a.z + u * b.z + v * c.z,
            w * a.w + u * b.w + v * c.w};
}

RT_HOST_DEVICE RT_FORCE_INLINE bool finite(Float2 value) {
    return packed_intersector::finite(value.x) &&
           packed_intersector::finite(value.y);
}

RT_HOST_DEVICE RT_FORCE_INLINE bool finite(Float3 value) {
    return packed_intersector::finite(value.x) &&
           packed_intersector::finite(value.y) &&
           packed_intersector::finite(value.z);
}

RT_HOST_DEVICE RT_FORCE_INLINE bool finite(Float4 value) {
    return packed_intersector::finite(value.x) &&
           packed_intersector::finite(value.y) &&
           packed_intersector::finite(value.z) &&
           packed_intersector::finite(value.w);
}

RT_HOST_DEVICE RT_FORCE_INLINE bool finite(
    const PackedSurfaceInteraction &surface) {
    return finite(surface.position) && finite(surface.geometric_normal) &&
           finite(surface.shading_normal) && finite(surface.dpdu) &&
           finite(surface.dpdv) && finite(surface.uv) &&
           finite(surface.vertex_color) &&
           packed_intersector::finite(surface.vertex_alpha);
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedShadingStatus reconstruct_triangle(
    const CompiledSceneView &scene, const PackedRay &ray,
    const PackedHit &hit, const PackedInstance &instance,
    const PackedTransform &transform, PackedSurfaceInteraction &surface) {
    if (instance.geometry_index >= scene.meshes.count ||
        hit.primitive_id >= scene.triangles.count) {
        return PackedShadingStatus::InvalidInput;
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
        if (absolute(determinant) > 1e-10f) {
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
    surface.emitter_id =
        resolve_emitter(scene, instance, triangle.material_slot);
    surface.primitive_id = triangle.primitive_id;
    if (surface.material_id >= scene.materials.count) {
        return PackedShadingStatus::InvalidMaterial;
    }
    return finite(surface) ? PackedShadingStatus::Success
                           : PackedShadingStatus::NonFinite;
}

RT_HOST_DEVICE RT_FORCE_INLINE void sphere_uv(Float3 normal, Float2 &uv) {
    constexpr float pi_f = 3.14159265358979323846f;
    const float clamped_y =
        maximum(-1.0f, minimum(1.0f, -normal.y));
    const float theta = ::acosf(clamped_y);
    const float phi = ::atan2f(-normal.z, normal.x) + pi_f;
    uv = {phi / (2.0f * pi_f), theta / pi_f};
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedShadingStatus reconstruct_sphere(
    const CompiledSceneView &scene, const PackedRay &ray,
    const PackedHit &hit, const PackedInstance &instance,
    const PackedTransform &transform, PackedSurfaceInteraction &surface) {
    Float3 center;
    float radius = 0.0f;
    std::uint32_t sphere_flags = 0;
    if (instance.geometry_type == PackedGeometryType::Sphere) {
        if (instance.geometry_index >= scene.spheres.count) {
            return PackedShadingStatus::InvalidInput;
        }
        const PackedSphere &sphere = scene.spheres[instance.geometry_index];
        center = sphere.center;
        radius = sphere.radius;
        sphere_flags = sphere.flags;
    } else {
        if (instance.geometry_index >= scene.moving_spheres.count) {
            return PackedShadingStatus::InvalidInput;
        }
        const PackedMovingSphere &sphere =
            scene.moving_spheres[instance.geometry_index];
        center = moving_sphere_center(sphere, ray.time);
        radius = sphere.radius;
        sphere_flags = sphere.flags;
    }
    if (!(radius > 0.0f)) {
        return PackedShadingStatus::InvalidInput;
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
    const float sin_theta = ::sinf(theta);
    const float cos_theta = ::cosf(theta);
    const float sin_alpha = ::sinf(alpha);
    const float cos_alpha = ::cosf(alpha);
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
    surface.emitter_id = resolve_emitter(scene, instance, 0);
    surface.primitive_id = instance.source_object_id;
    if (surface.material_id >= scene.materials.count) {
        return PackedShadingStatus::InvalidMaterial;
    }
    return finite(surface) ? PackedShadingStatus::Success
                           : PackedShadingStatus::NonFinite;
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedShadingStatus
reconstruct_compiled_hit_core(const CompiledSceneView &scene,
                              const PackedRay &ray, const PackedHit &hit,
                              PackedSurfaceInteraction &surface) {
    if (hit.instance_id >= scene.instances.count ||
        !packed_intersector::finite(hit.t)) {
        return PackedShadingStatus::InvalidInput;
    }
    const PackedInstance &instance = scene.instances[hit.instance_id];
    if (instance.transform_id >= scene.transforms.count) {
        return PackedShadingStatus::InvalidInput;
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
        if (instance.geometry_index >= scene.media.count) {
            return PackedShadingStatus::InvalidInput;
        }
        surface.position = ray_at(ray, hit.t);
        surface.geometric_normal = {1.0f, 0.0f, 0.0f};
        surface.shading_normal = surface.geometric_normal;
        surface.dpdu = {0.0f, 1.0f, 0.0f};
        surface.dpdv = {0.0f, 0.0f, 1.0f};
        surface.material_id =
            scene.media[instance.geometry_index].phase_material;
        surface.primitive_id = instance.source_object_id;
        surface.flags = PACKED_HIT_MEDIUM | PACKED_HIT_FRONT_FACE;
        if (surface.material_id >= scene.materials.count) {
            return PackedShadingStatus::InvalidMaterial;
        }
        return finite(surface) ? PackedShadingStatus::Success
                               : PackedShadingStatus::NonFinite;
    }
    return PackedShadingStatus::InvalidInput;
}

} // namespace packed_reconstruction

#endif

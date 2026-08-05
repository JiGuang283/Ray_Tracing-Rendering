#ifndef PACKED_LIGHT_CORE_H
#define PACKED_LIGHT_CORE_H

#include "flat_intersector_core.h"
#include "packed_material_core.h"
#include "packed_texture_core.h"
#include "rng.h"

#include <cfloat>
#include <cmath>
#include <cstdint>

namespace packed_light {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kRayEpsilon = 1e-4f;

namespace math {

RT_HOST_DEVICE RT_FORCE_INLINE float minimum(float a, float b) {
    return a < b ? a : b;
}

RT_HOST_DEVICE RT_FORCE_INLINE float maximum(float a, float b) {
    return a > b ? a : b;
}

RT_HOST_DEVICE RT_FORCE_INLINE float clamp(float value, float lower,
                                           float upper) {
    return maximum(lower, minimum(value, upper));
}

RT_HOST_DEVICE RT_FORCE_INLINE float absolute(float value) {
    return value < 0.0f ? -value : value;
}

RT_HOST_DEVICE RT_FORCE_INLINE bool finite(float value) {
    return value == value && value <= FLT_MAX && value >= -FLT_MAX;
}

RT_HOST_DEVICE RT_FORCE_INLINE bool finite(Float3 value) {
    return finite(value.x) && finite(value.y) && finite(value.z);
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

RT_HOST_DEVICE RT_FORCE_INLINE float dot(Float3 a, Float3 b) {
    return ::fmaf(a.x, b.x, ::fmaf(a.y, b.y, a.z * b.z));
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 cross(Float3 a, Float3 b) {
    return {::fmaf(a.y, b.z, -a.z * b.y),
            ::fmaf(a.z, b.x, -a.x * b.z),
            ::fmaf(a.x, b.y, -a.y * b.x)};
}

RT_HOST_DEVICE RT_FORCE_INLINE float length_squared(Float3 value) {
    return dot(value, value);
}

RT_HOST_DEVICE RT_FORCE_INLINE float length(Float3 value) {
    return ::sqrtf(length_squared(value));
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 normalize(Float3 value) {
    const float magnitude = length(value);
    if (!(magnitude > 0.0f) || !finite(magnitude)) {
        return {};
    }
    return multiply(value, 1.0f / magnitude);
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 from(Float4 value) {
    return {value.x, value.y, value.z};
}

RT_HOST_DEVICE RT_FORCE_INLINE float luminance(Float3 value) {
    return ::fmaf(0.2126f, value.x,
                  ::fmaf(0.7152f, value.y, 0.0722f * value.z));
}

} // namespace math

RT_HOST_DEVICE RT_FORCE_INLINE float unit_random(float value) {
    return math::clamp(value, 0.0f, 0.99999994f);
}

RT_HOST_DEVICE RT_FORCE_INLINE bool valid_range(Range32 range,
                                                std::uint32_t size) {
    return range.offset <= size && range.count <= size - range.offset;
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedLightStatus initialize_sample(
    const CompiledSceneView &scene, std::uint32_t light_id,
    PackedLightSample &sample) {
    sample = {};
    if (light_id >= scene.lights.count) {
        return PackedLightStatus::InvalidInput;
    }
    sample.light_id = light_id;
    sample.flags = scene.lights[light_id].flags;
    return PackedLightStatus::Success;
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedLightStatus finish_sample(
    PackedLightSample &sample) {
    if (!(sample.pdf > 0.0f) || !(sample.distance > 0.0f) ||
        !math::finite(sample.wi) || !math::finite(sample.radiance) ||
        !math::finite(sample.pdf) ||
        (!sample.is_infinite() && !math::finite(sample.distance))) {
        return PackedLightStatus::NoSample;
    }
    return PackedLightStatus::Success;
}

RT_HOST_DEVICE RT_FORCE_INLINE bool is_double_sided(
    const PackedLight &light) {
    return (light.flags & PACKED_LIGHT_DOUBLE_SIDED) != 0;
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedLightStatus evaluate_emission(
    const CompiledSceneView &scene, const PackedLight &light,
    Float3 position, Float2 uv, Float4 vertex_color, Float3 &emission) {
    PackedTextureEvalContext context;
    context.position = position;
    context.uv0 = uv;
    context.vertex_color = vertex_color;
    const PackedShadingStatus status =
        packed_material::evaluate_packed_material_emission_core(
            scene, light.material_id, context, emission);
    if (status == PackedShadingStatus::Success) {
        return PackedLightStatus::Success;
    }
    if (status == PackedShadingStatus::TextureFailure ||
        status == PackedShadingStatus::TextureStackOverflow) {
        return PackedLightStatus::TextureFailure;
    }
    return status == PackedShadingStatus::NonFinite
               ? PackedLightStatus::NonFinite
               : PackedLightStatus::InvalidInput;
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedLightStatus sample_point(
    const PackedLight &light, Float3 origin, PackedLightSample &sample) {
    const Float3 displacement =
        math::subtract(math::from(light.data0), origin);
    const float distance_squared = math::length_squared(displacement);
    if (!(distance_squared > 1e-12f) ||
        !math::finite(distance_squared)) {
        return PackedLightStatus::NoSample;
    }
    sample.distance = ::sqrtf(distance_squared);
    sample.wi = math::multiply(displacement, 1.0f / sample.distance);
    sample.radiance =
        math::multiply(math::from(light.radiance), 1.0f / distance_squared);
    sample.pdf = 1.0f;
    return finish_sample(sample);
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedLightStatus sample_directional(
    const PackedLight &light, PackedLightSample &sample) {
    sample.wi = math::multiply(math::normalize(math::from(light.data0)),
                               -1.0f);
    sample.distance = FLT_MAX;
    sample.radiance = math::from(light.radiance);
    sample.pdf = 1.0f;
    return finish_sample(sample);
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedLightStatus sample_spot(
    const PackedLight &light, Float3 origin, PackedLightSample &sample) {
    const PackedLightStatus status = sample_point(light, origin, sample);
    if (status != PackedLightStatus::Success) {
        return status;
    }
    const Float3 direction = math::normalize(math::from(light.data1));
    const float cosine = math::dot(math::multiply(sample.wi, -1.0f),
                                   direction);
    if (cosine < light.data0.w) {
        sample.radiance = {};
    }
    return PackedLightStatus::Success;
}

RT_HOST_DEVICE RT_FORCE_INLINE bool quad_geometry(
    const PackedLight &light, Float3 &q, Float3 &u, Float3 &v,
    Float3 &normal, float &area) {
    q = math::from(light.data0);
    u = math::from(light.data1);
    v = math::from(light.data2);
    const Float3 unnormalized = math::cross(u, v);
    area = math::length(unnormalized);
    normal = math::normalize(unnormalized);
    return area > 0.0f && math::finite(area) && math::finite(normal);
}

RT_HOST_DEVICE RT_FORCE_INLINE bool solve_quad_coordinates(
    Float3 displacement, Float3 u, Float3 v, float &alpha, float &beta) {
    const float uu = math::dot(u, u);
    const float uv = math::dot(u, v);
    const float vv = math::dot(v, v);
    const float du = math::dot(displacement, u);
    const float dv = math::dot(displacement, v);
    const float determinant = uu * vv - uv * uv;
    if (!(determinant > 1e-16f) || !math::finite(determinant)) {
        return false;
    }
    alpha = (du * vv - dv * uv) / determinant;
    beta = (dv * uu - du * uv) / determinant;
    return alpha >= 0.0f && alpha <= 1.0f && beta >= 0.0f && beta <= 1.0f;
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedLightStatus sample_quad(
    const PackedLight &light, Float3 origin, Float2 random,
    PackedLightSample &sample) {
    Float3 q;
    Float3 u;
    Float3 v;
    Float3 normal;
    float area = 0.0f;
    if (!quad_geometry(light, q, u, v, normal, area)) {
        return PackedLightStatus::InvalidInput;
    }
    const Float3 point = math::add(
        q, math::add(math::multiply(u, unit_random(random.x)),
                     math::multiply(v, unit_random(random.y))));
    const Float3 displacement = math::subtract(point, origin);
    const float distance_squared = math::length_squared(displacement);
    if (!(distance_squared > 1e-12f) ||
        !math::finite(distance_squared)) {
        return PackedLightStatus::NoSample;
    }
    sample.distance = ::sqrtf(distance_squared);
    sample.wi = math::multiply(displacement, 1.0f / sample.distance);
    const float signed_cosine =
        math::dot(math::multiply(sample.wi, -1.0f), normal);
    const float cosine = is_double_sided(light)
                             ? math::absolute(signed_cosine)
                             : signed_cosine;
    if (!(cosine > 0.0f)) {
        return PackedLightStatus::NoSample;
    }
    sample.radiance = math::from(light.radiance);
    sample.pdf = distance_squared / (area * cosine);
    return finish_sample(sample);
}

RT_HOST_DEVICE RT_FORCE_INLINE float quad_pdf(const PackedLight &light,
                                               Float3 origin,
                                               Float3 direction) {
    Float3 q;
    Float3 u;
    Float3 v;
    Float3 normal;
    float area = 0.0f;
    if (!quad_geometry(light, q, u, v, normal, area)) {
        return 0.0f;
    }
    const float direction_length = math::length(direction);
    if (!(direction_length > 0.0f)) {
        return 0.0f;
    }
    const float denominator = math::dot(direction, normal);
    if ((!is_double_sided(light) && denominator >= -1e-6f) ||
        (is_double_sided(light) &&
         math::absolute(denominator) <= 1e-6f)) {
        return 0.0f;
    }
    const float t = math::dot(math::subtract(q, origin), normal) /
                    denominator;
    if (!(t >= kRayEpsilon) || !math::finite(t)) {
        return 0.0f;
    }
    const Float3 point =
        math::add(origin, math::multiply(direction, t));
    float alpha = 0.0f;
    float beta = 0.0f;
    if (!solve_quad_coordinates(math::subtract(point, q), u, v, alpha,
                                beta)) {
        return 0.0f;
    }
    float cosine = -denominator / direction_length;
    if (is_double_sided(light)) {
        cosine = math::absolute(cosine);
    }
    if (!(cosine > 0.0f)) {
        return 0.0f;
    }
    const float distance_squared =
        t * t * direction_length * direction_length;
    return distance_squared / (area * cosine);
}

struct TriangleGeometry {
    Float3 object0{};
    Float3 object1{};
    Float3 object2{};
    Float3 world0{};
    Float3 world1{};
    Float3 world2{};
    Float3 normal{};
    float area = 0.0f;
    const PackedTriangle *triangle = nullptr;
    const PackedMesh *mesh = nullptr;
    const PackedInstance *instance = nullptr;
    const PackedTransform *transform = nullptr;
};

RT_HOST_DEVICE RT_FORCE_INLINE bool triangle_geometry(
    const CompiledSceneView &scene, const PackedLight &light,
    std::uint32_t triangle_id, TriangleGeometry &geometry) {
    if (light.instance_id >= scene.instances.count ||
        triangle_id >= scene.triangles.count) {
        return false;
    }
    const PackedInstance &instance = scene.instances[light.instance_id];
    if (instance.geometry_type != PackedGeometryType::Mesh ||
        instance.geometry_index >= scene.meshes.count ||
        instance.transform_id >= scene.transforms.count) {
        return false;
    }
    const PackedMesh &mesh = scene.meshes[instance.geometry_index];
    if (triangle_id < mesh.triangles.offset ||
        triangle_id >= mesh.triangles.offset + mesh.triangles.count) {
        return false;
    }
    const PackedTriangle &triangle = scene.triangles[triangle_id];
    if (triangle.vertex0 >= mesh.vertices.count ||
        triangle.vertex1 >= mesh.vertices.count ||
        triangle.vertex2 >= mesh.vertices.count) {
        return false;
    }
    const PackedTransform &transform = scene.transforms[instance.transform_id];
    const Float4 packed0 =
        scene.positions[mesh.vertices.offset + triangle.vertex0];
    const Float4 packed1 =
        scene.positions[mesh.vertices.offset + triangle.vertex1];
    const Float4 packed2 =
        scene.positions[mesh.vertices.offset + triangle.vertex2];
    geometry.object0 = math::from(packed0);
    geometry.object1 = math::from(packed1);
    geometry.object2 = math::from(packed2);
    geometry.world0 = packed_intersector::transform_point(
        transform.object_to_world, geometry.object0);
    geometry.world1 = packed_intersector::transform_point(
        transform.object_to_world, geometry.object1);
    geometry.world2 = packed_intersector::transform_point(
        transform.object_to_world, geometry.object2);
    const Float3 edge1 = math::subtract(geometry.world1, geometry.world0);
    const Float3 edge2 = math::subtract(geometry.world2, geometry.world0);
    Float3 unnormalized = math::cross(edge1, edge2);
    if ((instance.flags & PACKED_INSTANCE_FLIP_FACE) != 0) {
        unnormalized = math::multiply(unnormalized, -1.0f);
    }
    if ((triangle.flags & PACKED_TRIANGLE_REVERSE_EMITTER_NORMAL) != 0) {
        unnormalized = math::multiply(unnormalized, -1.0f);
    }
    geometry.area = 0.5f * math::length(unnormalized);
    geometry.normal = math::normalize(unnormalized);
    geometry.triangle = &triangle;
    geometry.mesh = &mesh;
    geometry.instance = &instance;
    geometry.transform = &transform;
    return geometry.area > 0.0f && math::finite(geometry.area) &&
           math::finite(geometry.normal);
}

RT_HOST_DEVICE RT_FORCE_INLINE Float2 triangle_uv(
    const CompiledSceneView &scene, const TriangleGeometry &geometry,
    float b0, float b1, float b2) {
    if ((geometry.triangle->flags & PACKED_TRIANGLE_HAS_UV) == 0) {
        return {b1, b2};
    }
    const std::uint32_t offset = geometry.mesh->vertices.offset;
    const Float2 uv0 = scene.uv0[offset + geometry.triangle->vertex0];
    const Float2 uv1 = scene.uv0[offset + geometry.triangle->vertex1];
    const Float2 uv2 = scene.uv0[offset + geometry.triangle->vertex2];
    return {b0 * uv0.x + b1 * uv1.x + b2 * uv2.x,
            b0 * uv0.y + b1 * uv1.y + b2 * uv2.y};
}

RT_HOST_DEVICE RT_FORCE_INLINE Float4 triangle_color(
    const CompiledSceneView &scene, const TriangleGeometry &geometry,
    float b0, float b1, float b2) {
    if ((geometry.triangle->flags & PACKED_TRIANGLE_HAS_COLOR) == 0) {
        return {1.0f, 1.0f, 1.0f, 1.0f};
    }
    const std::uint32_t offset = geometry.mesh->vertices.offset;
    const Float4 c0 =
        scene.vertex_colors[offset + geometry.triangle->vertex0];
    const Float4 c1 =
        scene.vertex_colors[offset + geometry.triangle->vertex1];
    const Float4 c2 =
        scene.vertex_colors[offset + geometry.triangle->vertex2];
    return {b0 * c0.x + b1 * c1.x + b2 * c2.x,
            b0 * c0.y + b1 * c1.y + b2 * c2.y,
            b0 * c0.z + b1 * c1.z + b2 * c2.z,
            b0 * c0.w + b1 * c1.w + b2 * c2.w};
}

RT_HOST_DEVICE RT_FORCE_INLINE bool choose_mesh_element(
    const CompiledSceneView &scene, const PackedLight &light, float random,
    std::uint32_t &element_index, float &local_random,
    float &element_probability) {
    const std::uint32_t count = light.element_indices.count;
    if (count == 0 || light.distribution.count != count * 2u ||
        !valid_range(light.element_indices,
                     scene.light_element_indices.count) ||
        !valid_range(light.distribution,
                     scene.light_distributions.count)) {
        return false;
    }
    const float target = unit_random(random);
    const std::uint32_t cdf_offset = light.distribution.offset + count;
    std::uint32_t lower = 0;
    std::uint32_t upper = count;
    while (lower < upper) {
        const std::uint32_t middle = lower + (upper - lower) / 2;
        if (scene.light_distributions[cdf_offset + middle] < target) {
            lower = middle + 1;
        } else {
            upper = middle;
        }
    }
    element_index = lower < count ? lower : count - 1;
    const float previous =
        element_index == 0
            ? 0.0f
            : scene.light_distributions[cdf_offset + element_index - 1];
    const float current =
        scene.light_distributions[cdf_offset + element_index];
    const float width = current - previous;
    local_random = width > 0.0f ? (target - previous) / width : 0.0f;
    local_random = unit_random(local_random);
    element_probability = scene.light_distributions[
        light.distribution.offset + element_index];
    return element_probability > 0.0f &&
           math::finite(element_probability);
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedLightStatus sample_mesh(
    const CompiledSceneView &scene, const PackedLight &light, Float3 origin,
    Float2 random, PackedLightSample &sample) {
    std::uint32_t element_index = 0;
    float local_random = 0.0f;
    float element_probability = 0.0f;
    if (!choose_mesh_element(scene, light, random.x, element_index,
                             local_random, element_probability)) {
        return PackedLightStatus::InvalidDistribution;
    }
    const std::uint32_t triangle_id = scene.light_element_indices[
        light.element_indices.offset + element_index];
    TriangleGeometry geometry;
    if (!triangle_geometry(scene, light, triangle_id, geometry)) {
        return PackedLightStatus::InvalidInput;
    }
    const float root = ::sqrtf(local_random);
    const float b0 = 1.0f - root;
    const float b1 = unit_random(random.y) * root;
    const float b2 = 1.0f - b0 - b1;
    const Float3 point = math::add(
        math::multiply(geometry.world0, b0),
        math::add(math::multiply(geometry.world1, b1),
                  math::multiply(geometry.world2, b2)));
    const Float3 displacement = math::subtract(point, origin);
    const float distance_squared = math::length_squared(displacement);
    if (!(distance_squared > 1e-12f) ||
        !math::finite(distance_squared)) {
        return PackedLightStatus::NoSample;
    }
    sample.distance = ::sqrtf(distance_squared);
    sample.wi = math::multiply(displacement, 1.0f / sample.distance);
    const float signed_cosine = math::dot(
        math::multiply(sample.wi, -1.0f), geometry.normal);
    const float cosine = is_double_sided(light)
                             ? math::absolute(signed_cosine)
                             : signed_cosine;
    if (!(cosine > 0.0f)) {
        return PackedLightStatus::NoSample;
    }
    const PackedLightStatus emission_status = evaluate_emission(
        scene, light, point, triangle_uv(scene, geometry, b0, b1, b2),
        triangle_color(scene, geometry, b0, b1, b2), sample.radiance);
    if (emission_status != PackedLightStatus::Success) {
        return emission_status;
    }
    sample.pdf = element_probability / geometry.area *
                 distance_squared / cosine;
    sample.element_id = triangle_id;
    return finish_sample(sample);
}

RT_HOST_DEVICE RT_FORCE_INLINE bool intersect_triangle(
    const TriangleGeometry &geometry, Float3 origin, Float3 direction,
    float &t) {
    using namespace triangle_intersection;
    TriangleKernelHit<float> intersection;
    if (!intersect_triangle_kernel(
            {origin.x, origin.y, origin.z},
            {direction.x, direction.y, direction.z},
            {geometry.world0.x, geometry.world0.y, geometry.world0.z},
            {geometry.world1.x, geometry.world1.y, geometry.world1.z},
            {geometry.world2.x, geometry.world2.y, geometry.world2.z},
            kRayEpsilon, FLT_MAX, intersection)) {
        return false;
    }
    t = intersection.t;
    return true;
}

RT_HOST_DEVICE RT_FORCE_INLINE float mesh_element_pdf(
    const CompiledSceneView &scene, const PackedLight &light,
    std::uint32_t element_index, Float3 origin, Float3 direction,
    const PackedHit *known_hit = nullptr) {
    if (element_index >= light.element_indices.count ||
        light.distribution.count != light.element_indices.count * 2u) {
        return 0.0f;
    }
    const std::uint32_t triangle_id = scene.light_element_indices[
        light.element_indices.offset + element_index];
    TriangleGeometry geometry;
    if (!triangle_geometry(scene, light, triangle_id, geometry)) {
        return 0.0f;
    }
    float t = 0.0f;
    if (known_hit != nullptr && known_hit->primitive_id == triangle_id) {
        t = known_hit->t;
    } else if (!intersect_triangle(geometry, origin, direction, t)) {
        return 0.0f;
    }
    const float direction_length = math::length(direction);
    if (!(direction_length > 0.0f)) {
        return 0.0f;
    }
    float cosine = -math::dot(direction, geometry.normal) /
                   direction_length;
    if (is_double_sided(light)) {
        cosine = math::absolute(cosine);
    }
    if (!(cosine > 0.0f)) {
        return 0.0f;
    }
    const float probability = scene.light_distributions[
        light.distribution.offset + element_index];
    const float distance_squared =
        t * t * direction_length * direction_length;
    const float result = probability / geometry.area *
                         distance_squared / cosine;
    return math::finite(result) && result > 0.0f ? result : 0.0f;
}

RT_HOST_DEVICE RT_FORCE_INLINE float mesh_pdf(
    const CompiledSceneView &scene, const PackedLight &light, Float3 origin,
    Float3 direction) {
    float result = 0.0f;
    for (std::uint32_t index = 0; index < light.element_indices.count;
         ++index) {
        result += mesh_element_pdf(scene, light, index, origin, direction);
    }
    return math::finite(result) ? result : 0.0f;
}

RT_HOST_DEVICE RT_FORCE_INLINE bool find_mesh_element(
    const CompiledSceneView &scene, const PackedLight &light,
    std::uint32_t triangle_id, std::uint32_t &element_index) {
    std::uint32_t lower = 0;
    std::uint32_t upper = light.element_indices.count;
    while (lower < upper) {
        const std::uint32_t middle = lower + (upper - lower) / 2;
        const std::uint32_t value = scene.light_element_indices[
            light.element_indices.offset + middle];
        if (value < triangle_id) {
            lower = middle + 1;
        } else {
            upper = middle;
        }
    }
    if (lower >= light.element_indices.count ||
        scene.light_element_indices[light.element_indices.offset + lower] !=
            triangle_id) {
        return false;
    }
    element_index = lower;
    return true;
}

struct SphereGeometry {
    const PackedInstance *instance = nullptr;
    const PackedSphere *sphere = nullptr;
    const PackedTransform *transform = nullptr;
};

RT_HOST_DEVICE RT_FORCE_INLINE bool sphere_geometry(
    const CompiledSceneView &scene, const PackedLight &light,
    SphereGeometry &geometry) {
    if (light.instance_id >= scene.instances.count) {
        return false;
    }
    const PackedInstance &instance = scene.instances[light.instance_id];
    if (instance.geometry_type != PackedGeometryType::Sphere ||
        instance.geometry_index >= scene.spheres.count ||
        instance.transform_id >= scene.transforms.count) {
        return false;
    }
    geometry.instance = &instance;
    geometry.sphere = &scene.spheres[instance.geometry_index];
    geometry.transform = &scene.transforms[instance.transform_id];
    return geometry.sphere->radius > 0.0f;
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 sphere_sample_normal(Float2 random) {
    const float z = 1.0f - 2.0f * unit_random(random.x);
    const float radius =
        ::sqrtf(math::maximum(0.0f, 1.0f - z * z));
    const float phi = 2.0f * kPi * unit_random(random.y);
    return {radius * ::cosf(phi), radius * ::sinf(phi), z};
}

RT_HOST_DEVICE RT_FORCE_INLINE float sphere_area_jacobian(
    const PackedTransform &transform, Float3 position_normal) {
    const float determinant = math::absolute(
        packed_intersector::linear_determinant(transform));
    const Float3 inverse_transpose =
        packed_intersector::transform_normal(transform, position_normal);
    return determinant * math::length(inverse_transpose);
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 oriented_sphere_normal(
    const SphereGeometry &geometry, Float3 position_normal) {
    Float3 object_normal = position_normal;
    if ((geometry.sphere->flags & PACKED_SPHERE_FLIP_ORIENTATION) != 0) {
        object_normal = math::multiply(object_normal, -1.0f);
    }
    Float3 world = math::normalize(packed_intersector::transform_normal(
        *geometry.transform, object_normal));
    if (packed_intersector::linear_determinant(*geometry.transform) < 0.0f) {
        world = math::multiply(world, -1.0f);
    }
    if ((geometry.instance->flags & PACKED_INSTANCE_FLIP_FACE) != 0) {
        world = math::multiply(world, -1.0f);
    }
    return world;
}

RT_HOST_DEVICE RT_FORCE_INLINE void sphere_uv(Float3 normal, Float2 &uv) {
    const float theta =
        ::acosf(math::clamp(-normal.y, -1.0f, 1.0f));
    const float phi = ::atan2f(-normal.z, normal.x) + kPi;
    uv = {phi / (2.0f * kPi), theta / kPi};
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedLightStatus sample_sphere(
    const CompiledSceneView &scene, const PackedLight &light, Float3 origin,
    Float2 random, PackedLightSample &sample) {
    SphereGeometry geometry;
    if (!sphere_geometry(scene, light, geometry)) {
        return PackedLightStatus::InvalidInput;
    }
    const Float3 position_normal = sphere_sample_normal(random);
    const Float3 object_point = math::add(
        geometry.sphere->center,
        math::multiply(position_normal, geometry.sphere->radius));
    const Float3 world_point = packed_intersector::transform_point(
        geometry.transform->object_to_world, object_point);
    const Float3 world_normal =
        oriented_sphere_normal(geometry, position_normal);
    const Float3 displacement = math::subtract(world_point, origin);
    const float distance_squared = math::length_squared(displacement);
    const float jacobian =
        sphere_area_jacobian(*geometry.transform, position_normal);
    const float object_area = 4.0f * kPi * geometry.sphere->radius *
                              geometry.sphere->radius;
    if (!(distance_squared > 1e-12f) || !(jacobian > 0.0f) ||
        !(object_area > 0.0f)) {
        return PackedLightStatus::NoSample;
    }
    sample.distance = ::sqrtf(distance_squared);
    sample.wi = math::multiply(displacement, 1.0f / sample.distance);
    const float signed_cosine = math::dot(
        math::multiply(sample.wi, -1.0f), world_normal);
    const float cosine = is_double_sided(light)
                             ? math::absolute(signed_cosine)
                             : signed_cosine;
    if (!(cosine > 0.0f)) {
        return PackedLightStatus::NoSample;
    }
    Float3 texture_normal = position_normal;
    if ((geometry.sphere->flags & PACKED_SPHERE_FLIP_ORIENTATION) != 0) {
        texture_normal = math::multiply(texture_normal, -1.0f);
    }
    Float2 uv;
    sphere_uv(texture_normal, uv);
    const PackedLightStatus emission_status = evaluate_emission(
        scene, light, world_point, uv, {1.0f, 1.0f, 1.0f, 1.0f},
        sample.radiance);
    if (emission_status != PackedLightStatus::Success) {
        return emission_status;
    }
    const float area_pdf = 1.0f / (object_area * jacobian);
    sample.pdf = area_pdf * distance_squared / cosine;
    sample.element_id = light.instance_id;
    return finish_sample(sample);
}

RT_HOST_DEVICE RT_FORCE_INLINE float sphere_pdf(
    const CompiledSceneView &scene, const PackedLight &light, Float3 origin,
    Float3 direction, const PackedHit *known_hit = nullptr) {
    SphereGeometry geometry;
    if (!sphere_geometry(scene, light, geometry)) {
        return 0.0f;
    }
    PackedRay ray;
    ray.origin = origin;
    ray.direction = direction;
    ray.t_min = kRayEpsilon;
    ray.t_max = FLT_MAX;
    const PackedRay object_ray =
        packed_intersector::transform_ray_to_object(ray,
                                                    *geometry.transform);
    PackedHit hit{};
    float t = 0.0f;
    if (known_hit != nullptr && known_hit->instance_id == light.instance_id) {
        t = known_hit->t;
    } else if (packed_intersector::intersect_sphere(
                   geometry.sphere->center, geometry.sphere->radius,
                   object_ray, kRayEpsilon, FLT_MAX, hit)) {
        t = hit.t;
    } else {
        return 0.0f;
    }
    const Float3 object_point = packed_intersector::ray_at(object_ray, t);
    const Float3 position_normal = math::normalize(math::subtract(
        object_point, geometry.sphere->center));
    const Float3 world_normal =
        oriented_sphere_normal(geometry, position_normal);
    const float direction_length = math::length(direction);
    if (!(direction_length > 0.0f)) {
        return 0.0f;
    }
    float cosine = -math::dot(direction, world_normal) / direction_length;
    if (is_double_sided(light)) {
        cosine = math::absolute(cosine);
    }
    const float jacobian =
        sphere_area_jacobian(*geometry.transform, position_normal);
    const float object_area = 4.0f * kPi * geometry.sphere->radius *
                              geometry.sphere->radius;
    if (!(cosine > 0.0f) || !(jacobian > 0.0f) ||
        !(object_area > 0.0f)) {
        return 0.0f;
    }
    const float distance_squared =
        t * t * direction_length * direction_length;
    const float result = distance_squared /
                         (object_area * jacobian * cosine);
    return math::finite(result) && result > 0.0f ? result : 0.0f;
}

RT_HOST_DEVICE RT_FORCE_INLINE bool sample_cdf(
    const float *cdf, std::uint32_t count, float random,
    std::uint32_t &index, float &local_random) {
    if (cdf == nullptr || count == 0) {
        return false;
    }
    const float target = unit_random(random);
    std::uint32_t lower = 0;
    std::uint32_t upper = count;
    while (lower < upper) {
        const std::uint32_t middle = lower + (upper - lower) / 2;
        if (cdf[middle + 1] < target) {
            lower = middle + 1;
        } else {
            upper = middle;
        }
    }
    index = lower < count ? lower : count - 1;
    const float width = cdf[index + 1] - cdf[index];
    if (!(width > 0.0f)) {
        return false;
    }
    local_random = unit_random((target - cdf[index]) / width);
    return true;
}

struct EnvironmentLayout {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t conditional_probability = 0;
    std::uint32_t conditional_cdf = 0;
    std::uint32_t marginal_probability = 0;
    std::uint32_t marginal_cdf = 0;
};

RT_HOST_DEVICE RT_FORCE_INLINE bool environment_layout(
    const CompiledSceneView &scene, const PackedLight &light,
    EnvironmentLayout &layout) {
    layout.width = static_cast<std::uint32_t>(light.data0.x);
    layout.height = static_cast<std::uint32_t>(light.data0.y);
    if (layout.width == 0 || layout.height == 0 ||
        light.image_id >= scene.images.count ||
        !valid_range(light.distribution,
                     scene.light_distributions.count)) {
        return false;
    }
    const std::uint64_t conditional_count =
        static_cast<std::uint64_t>(layout.width) * layout.height;
    const std::uint64_t conditional_cdf_count =
        static_cast<std::uint64_t>(layout.height) * (layout.width + 1u);
    const std::uint64_t expected = conditional_count +
                                   conditional_cdf_count + layout.height +
                                   layout.height + 1u;
    if (expected != light.distribution.count) {
        return false;
    }
    layout.conditional_probability = light.distribution.offset;
    layout.conditional_cdf =
        layout.conditional_probability +
        static_cast<std::uint32_t>(conditional_count);
    layout.marginal_probability =
        layout.conditional_cdf +
        static_cast<std::uint32_t>(conditional_cdf_count);
    layout.marginal_cdf = layout.marginal_probability + layout.height;
    return true;
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 environment_texel(
    const CompiledSceneView &scene, const PackedLight &light, Float2 uv) {
    PackedTextureNode texture;
    texture.type = PackedTextureType::Image;
    texture.image_id = light.image_id;
    texture.sampler_flags =
        (1u << PACKED_SAMPLER_WRAP_V_SHIFT) |
        (1u << PACKED_SAMPLER_FILTER_SHIFT);
    if ((light.flags & PACKED_LIGHT_ENVIRONMENT_SRGB) != 0) {
        texture.sampler_flags |= PACKED_SAMPLER_SRGB;
    }
    const Float4 value =
        packed_texture::evaluate_image(scene, texture, uv);
    return {value.x, value.y, value.z};
}

RT_HOST_DEVICE RT_FORCE_INLINE bool environment_direction_to_uv(
    const PackedLight &light, Float3 direction, Float2 &uv,
    float &jacobian) {
    direction = math::normalize(direction);
    if (!math::finite(direction) || math::length_squared(direction) == 0.0f) {
        return false;
    }
    if ((light.flags & PACKED_LIGHT_ENVIRONMENT_PROBE) != 0) {
        const float planar = ::sqrtf(direction.x * direction.x +
                                     direction.y * direction.y);
        const float theta =
            ::acosf(math::clamp(direction.z, -1.0f, 1.0f));
        const float radial = theta / kPi;
        const float scale = planar > 0.0f ? radial / planar : 0.0f;
        uv.x = 0.5f * (direction.x * scale + 1.0f);
        uv.y = 1.0f - 0.5f * (direction.y * scale + 1.0f);
        const float sine = ::sinf(theta);
        jacobian = radial > 1e-7f
                       ? 4.0f * kPi * sine / radial
                       : 4.0f * kPi * kPi;
        return jacobian > 0.0f && math::finite(jacobian);
    }
    const float theta =
        ::acosf(math::clamp(direction.y, -1.0f, 1.0f));
    const float phi = ::atan2f(-direction.z, direction.x) + kPi;
    uv = {phi / (2.0f * kPi), theta / kPi};
    jacobian = 2.0f * kPi * kPi * ::sinf(theta);
    return jacobian > 1e-12f && math::finite(jacobian);
}

RT_HOST_DEVICE RT_FORCE_INLINE bool environment_uv_to_direction(
    const PackedLight &light, Float2 uv, Float3 &direction,
    float &jacobian) {
    if ((light.flags & PACKED_LIGHT_ENVIRONMENT_PROBE) != 0) {
        const float centered_x = 2.0f * uv.x - 1.0f;
        const float centered_y = 1.0f - 2.0f * uv.y;
        const float radial = ::sqrtf(centered_x * centered_x +
                                     centered_y * centered_y);
        if (radial > 1.0f) {
            return false;
        }
        const float theta = kPi * radial;
        const float phi = ::atan2f(centered_y, centered_x);
        const float sine = ::sinf(theta);
        direction = {sine * ::cosf(phi), sine * ::sinf(phi),
                     ::cosf(theta)};
        jacobian = radial > 1e-7f
                       ? 4.0f * kPi * sine / radial
                       : 4.0f * kPi * kPi;
        return jacobian > 0.0f && math::finite(jacobian);
    }
    const float phi = uv.x * 2.0f * kPi - kPi;
    const float theta = uv.y * kPi;
    const float sine = ::sinf(theta);
    direction = {sine * ::cosf(phi), ::cosf(theta),
                 -sine * ::sinf(phi)};
    jacobian = 2.0f * kPi * kPi * sine;
    return jacobian > 1e-12f && math::finite(jacobian);
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedLightStatus sample_environment(
    const CompiledSceneView &scene, const PackedLight &light, Float2 random,
    PackedLightSample &sample) {
    EnvironmentLayout layout;
    if (!environment_layout(scene, light, layout)) {
        return PackedLightStatus::InvalidDistribution;
    }
    const float *distribution = scene.light_distributions.data;
    std::uint32_t row = 0;
    float local_v = 0.0f;
    if (!sample_cdf(distribution + layout.marginal_cdf, layout.height,
                    random.y, row, local_v)) {
        return PackedLightStatus::NoSample;
    }
    std::uint32_t column = 0;
    float local_u = 0.0f;
    const std::uint32_t row_cdf =
        layout.conditional_cdf + row * (layout.width + 1u);
    if (!sample_cdf(distribution + row_cdf, layout.width, random.x,
                    column, local_u)) {
        return PackedLightStatus::NoSample;
    }
    const Float2 uv{(static_cast<float>(column) + local_u) / layout.width,
                    (static_cast<float>(row) + local_v) / layout.height};
    float jacobian = 0.0f;
    if (!environment_uv_to_direction(light, uv, sample.wi, jacobian)) {
        return PackedLightStatus::NoSample;
    }
    const float column_probability = distribution[
        layout.conditional_probability + row * layout.width + column];
    const float row_probability =
        distribution[layout.marginal_probability + row];
    const float uv_pdf = column_probability * row_probability *
                         layout.width * layout.height;
    sample.pdf = uv_pdf / jacobian;
    sample.distance = FLT_MAX;
    sample.radiance = environment_texel(scene, light, uv);
    sample.element_id = row * layout.width + column;
    return finish_sample(sample);
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedLightStatus environment_radiance_core(
    const CompiledSceneView &scene, const PackedLight &light,
    Float3 direction, Float3 &radiance) {
    EnvironmentLayout layout;
    if (!environment_layout(scene, light, layout)) {
        radiance = {};
        return PackedLightStatus::InvalidDistribution;
    }
    Float2 uv;
    float jacobian = 0.0f;
    if (!environment_direction_to_uv(light, direction, uv, jacobian)) {
        radiance = {};
        return PackedLightStatus::NoSample;
    }
    radiance = environment_texel(scene, light, uv);
    return math::finite(radiance) ? PackedLightStatus::Success
                                  : PackedLightStatus::NonFinite;
}

RT_HOST_DEVICE RT_FORCE_INLINE float environment_pdf(
    const CompiledSceneView &scene, const PackedLight &light,
    Float3 direction) {
    EnvironmentLayout layout;
    if (!environment_layout(scene, light, layout)) {
        return 0.0f;
    }
    Float2 uv;
    float jacobian = 0.0f;
    if (!environment_direction_to_uv(light, direction, uv, jacobian)) {
        return 0.0f;
    }
    const std::uint32_t column = math::minimum(
        static_cast<float>(layout.width - 1u),
        ::floorf(math::clamp(uv.x, 0.0f, 0.99999994f) * layout.width));
    const std::uint32_t row = math::minimum(
        static_cast<float>(layout.height - 1u),
        ::floorf(math::clamp(uv.y, 0.0f, 0.99999994f) * layout.height));
    const float column_probability = scene.light_distributions[
        layout.conditional_probability + row * layout.width + column];
    const float row_probability = scene.light_distributions[
        layout.marginal_probability + row];
    const float result = column_probability * row_probability *
                         layout.width * layout.height / jacobian;
    return math::finite(result) && result > 0.0f ? result : 0.0f;
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedLightStatus sample_packed_light_core(
    const CompiledSceneView &scene, std::uint32_t light_id, Float3 origin,
    Float2 random, PackedLightSample &sample) {
    PackedLightStatus status = initialize_sample(scene, light_id, sample);
    if (status != PackedLightStatus::Success) {
        return status;
    }
    if (!math::finite(origin)) {
        return PackedLightStatus::InvalidInput;
    }
    const PackedLight &light = scene.lights[light_id];
    switch (light.type) {
    case PackedLightType::Point:
        return sample_point(light, origin, sample);
    case PackedLightType::Directional:
        return sample_directional(light, sample);
    case PackedLightType::Spot:
        return sample_spot(light, origin, sample);
    case PackedLightType::Quad:
        return sample_quad(light, origin, random, sample);
    case PackedLightType::Environment:
        return sample_environment(scene, light, random, sample);
    case PackedLightType::SphereEmitter:
        return sample_sphere(scene, light, origin, random, sample);
    case PackedLightType::TriangleEmitter:
    case PackedLightType::MeshEmitter:
        return sample_mesh(scene, light, origin, random, sample);
    }
    return PackedLightStatus::InvalidInput;
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedLightStatus packed_light_pdf_core(
    const CompiledSceneView &scene, std::uint32_t light_id, Float3 origin,
    Float3 direction, float &pdf) {
    pdf = 0.0f;
    if (light_id >= scene.lights.count || !math::finite(origin) ||
        !math::finite(direction)) {
        return PackedLightStatus::InvalidInput;
    }
    const PackedLight &light = scene.lights[light_id];
    switch (light.type) {
    case PackedLightType::Quad:
        pdf = quad_pdf(light, origin, direction);
        break;
    case PackedLightType::Environment:
        pdf = environment_pdf(scene, light, direction);
        break;
    case PackedLightType::SphereEmitter:
        pdf = sphere_pdf(scene, light, origin, direction);
        break;
    case PackedLightType::TriangleEmitter:
    case PackedLightType::MeshEmitter:
        pdf = mesh_pdf(scene, light, origin, direction);
        break;
    case PackedLightType::Point:
    case PackedLightType::Directional:
    case PackedLightType::Spot:
        pdf = 0.0f;
        break;
    }
    return math::finite(pdf) && pdf >= 0.0f
               ? PackedLightStatus::Success
               : PackedLightStatus::NonFinite;
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedLightStatus
packed_emitter_hit_pdf_core(const CompiledSceneView &scene,
                            std::uint32_t light_id, Float3 origin,
                            Float3 direction, const PackedHit &hit,
                            float &pdf) {
    pdf = 0.0f;
    if (light_id >= scene.lights.count || !math::finite(origin) ||
        !math::finite(direction)) {
        return PackedLightStatus::InvalidInput;
    }
    const PackedLight &light = scene.lights[light_id];
    if ((light.flags & PACKED_LIGHT_BSDF_HITTABLE) == 0) {
        return PackedLightStatus::Success;
    }
    if (light.type == PackedLightType::SphereEmitter) {
        pdf = sphere_pdf(scene, light, origin, direction, &hit);
    } else if (light.type == PackedLightType::TriangleEmitter ||
               light.type == PackedLightType::MeshEmitter) {
        std::uint32_t element = 0;
        if (!find_mesh_element(scene, light, hit.primitive_id, element)) {
            return PackedLightStatus::InvalidInput;
        }
        pdf = mesh_element_pdf(scene, light, element, origin, direction,
                               &hit);
    } else {
        return packed_light_pdf_core(scene, light_id, origin, direction,
                                     pdf);
    }
    return math::finite(pdf) && pdf >= 0.0f
               ? PackedLightStatus::Success
               : PackedLightStatus::NonFinite;
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedLightStatus
sample_non_delta_light_core(const CompiledSceneView &scene, Float3 origin,
                            RNG &rng, SelectedPackedLightSample &selected) {
    selected = {};
    const std::uint32_t count = scene.non_delta_light_indices.count;
    if (count == 0) {
        return PackedLightStatus::NoSample;
    }
    if (scene.light_selection_probabilities.count != count ||
        scene.light_cdf.count != count) {
        return PackedLightStatus::InvalidDistribution;
    }
    const float selection_random = static_cast<float>(rng.next());
    const float sample_u = static_cast<float>(rng.next());
    const float sample_v = static_cast<float>(rng.next());
    const float target = unit_random(selection_random);
    std::uint32_t lower = 0;
    std::uint32_t upper = count;
    while (lower < upper) {
        const std::uint32_t middle = lower + (upper - lower) / 2;
        if (scene.light_cdf[middle] < target) {
            lower = middle + 1;
        } else {
            upper = middle;
        }
    }
    const std::uint32_t index = lower < count ? lower : count - 1;
    const std::uint32_t light_id = scene.non_delta_light_indices[index];
    const float probability = scene.light_selection_probabilities[index];
    if (light_id >= scene.lights.count || !(probability > 0.0f) ||
        !math::finite(probability)) {
        return PackedLightStatus::InvalidDistribution;
    }
    selected.selection_index = index;
    selected.selection_probability = probability;
    return sample_packed_light_core(scene, light_id, origin,
                                    {sample_u, sample_v}, selected.sample);
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedLightStatus light_sampler_pdf_core(
    const CompiledSceneView &scene, Float3 origin, Float3 direction,
    float &pdf) {
    pdf = 0.0f;
    const std::uint32_t count = scene.non_delta_light_indices.count;
    if (scene.light_selection_probabilities.count != count) {
        return PackedLightStatus::InvalidDistribution;
    }
    for (std::uint32_t index = 0; index < count; ++index) {
        const std::uint32_t light_id = scene.non_delta_light_indices[index];
        if (light_id >= scene.lights.count) {
            return PackedLightStatus::InvalidDistribution;
        }
        const PackedLight &light = scene.lights[light_id];
        if ((light.flags & PACKED_LIGHT_BSDF_HITTABLE) == 0) {
            continue;
        }
        float conditional = 0.0f;
        const PackedLightStatus status = packed_light_pdf_core(
            scene, light_id, origin, direction, conditional);
        if (status != PackedLightStatus::Success) {
            return status;
        }
        pdf += scene.light_selection_probabilities[index] * conditional;
    }
    return math::finite(pdf) && pdf >= 0.0f
               ? PackedLightStatus::Success
               : PackedLightStatus::NonFinite;
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedLightStatus emitter_hit_mis_pdf_core(
    const CompiledSceneView &scene, std::uint32_t light_id, Float3 origin,
    Float3 direction, const PackedHit &hit, float &pdf) {
    pdf = 0.0f;
    if (light_id >= scene.lights.count) {
        return PackedLightStatus::InvalidInput;
    }
    const PackedLight &light = scene.lights[light_id];
    float conditional = 0.0f;
    const PackedLightStatus status = packed_emitter_hit_pdf_core(
        scene, light_id, origin, direction, hit, conditional);
    if (status != PackedLightStatus::Success) {
        return status;
    }
    pdf = light.selection_probability * conditional;
    return math::finite(pdf) && pdf >= 0.0f
               ? PackedLightStatus::Success
               : PackedLightStatus::NonFinite;
}

RT_HOST_DEVICE RT_FORCE_INLINE std::uint32_t environment_light_id(
    const CompiledSceneView &scene) {
    for (std::uint32_t index = 0; index < scene.lights.count; ++index) {
        if (scene.lights[index].type == PackedLightType::Environment) {
            return index;
        }
    }
    return kInvalidPackedIndex;
}

} // namespace packed_light

#endif

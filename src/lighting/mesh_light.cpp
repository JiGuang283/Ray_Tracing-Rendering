#include "mesh_light.h"

#include "mesh_instance.h"
#include "shading/shader_context.h"
#include "shading/shading.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

MeshLight::MeshLight(std::shared_ptr<const MeshInstance> instance,
                     std::uint32_t material_slot, bool flip_orientation)
    : m_instance(std::move(instance)), m_material_slot(material_slot) {
    if (!m_instance) {
        throw std::invalid_argument("MeshLight requires a mesh instance.");
    }
    if (m_material_slot >= m_instance->materials().size()) {
        throw std::invalid_argument(
            "MeshLight material slot is out of range.");
    }
    m_double_sided =
        m_instance->materials()[m_material_slot]->is_double_sided();

    const MeshAsset &asset = *m_instance->asset();
    const Transform &transform = m_instance->object_to_world();
    m_triangles.reserve(asset.triangle_count());
    m_cdf.reserve(asset.triangle_count());
    for (std::uint32_t index = 0; index < asset.triangle_count(); ++index) {
        const MeshTriangle &source = asset.triangle(index);
        if (source.material_slot != m_material_slot) {
            continue;
        }
        TriangleEntry entry;
        entry.triangle_index = index;
        if (flip_orientation) {
            entry.corner_order = {0, 2, 1};
        }
        const MeshVertex &v0 =
            asset.vertex(source.vertices[entry.corner_order[0]]);
        const MeshVertex &v1 =
            asset.vertex(source.vertices[entry.corner_order[1]]);
        const MeshVertex &v2 =
            asset.vertex(source.vertices[entry.corner_order[2]]);
        entry.v0 = transform.point_to_world(v0.position);
        entry.v1 = transform.point_to_world(v1.position);
        entry.v2 = transform.point_to_world(v2.position);
        entry.edge1 = entry.v1 - entry.v0;
        entry.edge2 = entry.v2 - entry.v0;
        const vec3 normal = cross(entry.edge1, entry.edge2);
        entry.area = 0.5 * normal.length();
        if (entry.area <= 0.0) {
            continue;
        }
        entry.normal = unit_vector(normal);
        m_total_area += entry.area;
        m_triangles.push_back(entry);
        m_cdf.push_back(m_total_area);
    }
}

LightSample MeshLight::sample(const point3 &p, const vec2 &u) const {
    LightSample sample;
    sample.Li = color(0, 0, 0);
    sample.wi = vec3(0, 0, 1);
    sample.pdf = 0.0;
    sample.dist = infinity;
    sample.is_delta = false;
    if (m_triangles.empty() || m_total_area <= 0.0) {
        return sample;
    }

    double local_u = 0.0;
    const TriangleEntry &triangle = choose_triangle(u.x(), local_u);
    const double root = std::sqrt(local_u);
    const double b0 = 1.0 - root;
    const double b1 = u.y() * root;
    const double b2 = 1.0 - b0 - b1;
    const point3 light_point =
        b0 * triangle.v0 + b1 * triangle.v1 + b2 * triangle.v2;

    const vec3 displacement = light_point - p;
    const double distance_squared = displacement.length_squared();
    if (distance_squared <= 0.0) {
        return sample;
    }
    sample.dist = std::sqrt(distance_squared);
    sample.wi = displacement / sample.dist;
    const double signed_cosine = dot(-sample.wi, triangle.normal);
    const double cosine =
        m_double_sided ? std::abs(signed_cosine) : signed_cosine;
    if (cosine <= 0.0) {
        return sample;
    }

    sample.Li = evaluate_emission(triangle, b0, b1, b2,
                                  signed_cosine > 0.0);
    sample.pdf = distance_squared / (m_total_area * cosine);
    return sample;
}

double MeshLight::pdf(const point3 &origin, const vec3 &direction) const {
    if (m_triangles.empty() || m_total_area <= 0.0) {
        return 0.0;
    }

    double result = 0.0;
    const double direction_length_squared = direction.length_squared();
    if (direction_length_squared <= 0.0) {
        return 0.0;
    }
    const vec3 unit_direction = direction / std::sqrt(direction_length_squared);
    for (const TriangleEntry &triangle : m_triangles) {
        double t = 0.0;
        if (!intersect_triangle(triangle, origin, direction, t)) {
            continue;
        }
        double cosine = -dot(unit_direction, triangle.normal);
        if (m_double_sided) {
            cosine = std::abs(cosine);
        }
        if (cosine > 0.0) {
            result += t * t * direction_length_squared /
                      (m_total_area * cosine);
        }
    }
    return result;
}

color MeshLight::power() const {
    const double side_factor = m_double_sided ? 2.0 : 1.0;
    return side_factor * pi * m_total_area *
           m_instance->materials()[m_material_slot]->emission_estimate();
}

bool MeshLight::is_bsdf_hittable() const {
    return true;
}

const MeshLight::TriangleEntry &
MeshLight::choose_triangle(double u, double &local_u) const {
    const double target = clamp(u, 0.0, 0.999999999) * m_total_area;
    auto found = std::lower_bound(m_cdf.begin(), m_cdf.end(), target);
    std::size_t index = static_cast<std::size_t>(found - m_cdf.begin());
    index = std::min(index, m_triangles.size() - 1);
    const double previous = index == 0 ? 0.0 : m_cdf[index - 1];
    const double width = m_cdf[index] - previous;
    local_u = width > 0.0 ? (target - previous) / width : 0.0;
    local_u = clamp(local_u, 0.0, 0.999999999);
    return m_triangles[index];
}

color MeshLight::evaluate_emission(const TriangleEntry &triangle, double b0,
                                   double b1, double b2,
                                   bool front_face) const {
    const MeshAsset &asset = *m_instance->asset();
    const MeshTriangle &source = asset.triangle(triangle.triangle_index);
    const MeshVertex &v0 =
        asset.vertex(source.vertices[triangle.corner_order[0]]);
    const MeshVertex &v1 =
        asset.vertex(source.vertices[triangle.corner_order[1]]);
    const MeshVertex &v2 =
        asset.vertex(source.vertices[triangle.corner_order[2]]);

    ShaderEvalContext context;
    context.position =
        b0 * triangle.v0 + b1 * triangle.v1 + b2 * triangle.v2;
    const vec3 oriented_normal =
        front_face ? triangle.normal : -triangle.normal;
    context.geometry_normal = oriented_normal;
    context.shading_normal = oriented_normal;
    context.frame.build_from_tangent_space(
        oriented_normal, triangle.edge1, triangle.edge2);
    context.front_face = front_face;
    context.primitive_id = static_cast<int>(triangle.triangle_index);
    context.material_id = static_cast<int>(m_material_slot);
    if (has_mesh_attribute(source.attributes, MESH_ATTRIBUTE_UV0)) {
        context.uv0 = b0 * v0.uv0 + b1 * v1.uv0 + b2 * v2.uv0;
    }
    if (has_mesh_attribute(source.attributes, MESH_ATTRIBUTE_COLOR0)) {
        context.vertex_color =
            b0 * v0.color0 + b1 * v1.color0 + b2 * v2.color0;
        context.vertex_alpha = b0 * v0.color_alpha + b1 * v1.color_alpha +
                               b2 * v2.color_alpha;
    }

    ShaderScratch scratch;
    MaterialOutput output;
    m_instance->materials()[m_material_slot]->evaluate(context, scratch,
                                                       output);
    return output.has_emission ? output.emission : color(0, 0, 0);
}

bool MeshLight::intersect_triangle(const TriangleEntry &triangle,
                                   const point3 &origin,
                                   const vec3 &direction, double &t) {
    const vec3 pvec = cross(direction, triangle.edge2);
    const double determinant = dot(triangle.edge1, pvec);
    if (std::abs(determinant) < 1e-8) {
        return false;
    }
    const double inverse = 1.0 / determinant;
    const vec3 tvec = origin - triangle.v0;
    const double u = dot(tvec, pvec) * inverse;
    if (u < 0.0 || u > 1.0) {
        return false;
    }
    const vec3 qvec = cross(tvec, triangle.edge1);
    const double v = dot(direction, qvec) * inverse;
    if (v < 0.0 || u + v > 1.0) {
        return false;
    }
    t = dot(triangle.edge2, qvec) * inverse;
    return t >= 0.001;
}

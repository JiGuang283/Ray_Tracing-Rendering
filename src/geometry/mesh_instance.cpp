#include "mesh_instance.h"

#include <cmath>
#include <stdexcept>

MeshInstance::MeshInstance(std::shared_ptr<const MeshAsset> asset,
                           std::vector<MaterialHandle> materials,
                           Transform object_to_world)
    : m_asset(std::move(asset)), m_materials(std::move(materials)),
      m_object_to_world(std::move(object_to_world)) {
    if (!m_asset) {
        throw std::invalid_argument("MeshInstance requires a MeshAsset.");
    }
    if (m_materials.size() < m_asset->material_slot_count()) {
        throw std::invalid_argument(
            "MeshInstance material table does not cover every material slot.");
    }
    for (const MaterialHandle &material : m_materials) {
        if (!material) {
            throw std::invalid_argument(
                "MeshInstance material table contains a null material.");
        }
    }
    m_world_bounds = m_object_to_world.bounds_to_world(m_asset->bounds());
}

bool MeshInstance::hit(const ray &world_ray, double t_min, double t_max,
                       hit_record &record) const {
    MeshIntersection intersection;
    if (!m_asset->intersect(m_object_to_world.ray_to_object(world_ray), t_min,
                            t_max, intersection)) {
        return false;
    }
    populate_record(world_ray, intersection, record);
    return true;
}

bool MeshInstance::hit(const ray &world_ray, double t_min, double t_max,
                       hit_record &record, RNG & /*rng*/) const {
    return hit(world_ray, t_min, t_max, record);
}

bool MeshInstance::bounding_box(double /*time0*/, double /*time1*/,
                                aabb &output_box) const {
    output_box = m_world_bounds;
    return true;
}

void MeshInstance::populate_record(const ray &world_ray,
                                   const MeshIntersection &intersection,
                                   hit_record &record) const {
    const MeshTriangle &triangle =
        m_asset->triangle(intersection.triangle_index);
    const MeshVertex &v0 = m_asset->vertex(triangle.vertices[0]);
    const MeshVertex &v1 = m_asset->vertex(triangle.vertices[1]);
    const MeshVertex &v2 = m_asset->vertex(triangle.vertices[2]);
    const double u = intersection.barycentric_u;
    const double v = intersection.barycentric_v;
    const double w = 1.0 - u - v;

    const vec3 object_edge1 = v1.position - v0.position;
    const vec3 object_edge2 = v2.position - v0.position;
    const vec3 world_edge1 =
        m_object_to_world.vector_to_world(object_edge1);
    const vec3 world_edge2 =
        m_object_to_world.vector_to_world(object_edge2);
    const vec3 outward_normal = unit_vector(cross(world_edge1, world_edge2));

    vec3 object_shading_normal =
        unit_vector(cross(object_edge1, object_edge2));
    vec3 shading_normal = outward_normal;
    if (has_mesh_attribute(triangle.attributes, MESH_ATTRIBUTE_NORMAL)) {
        object_shading_normal =
            unit_vector(w * v0.normal + u * v1.normal + v * v2.normal);
        shading_normal =
            unit_vector(m_object_to_world.normal_to_world(
                object_shading_normal));
        if (dot(shading_normal, outward_normal) < 0.0) {
            shading_normal = -shading_normal;
        }
    }

    vec3 object_dpdu = object_edge1;
    vec3 object_dpdv = object_edge2;
    if (has_mesh_attribute(triangle.attributes, MESH_ATTRIBUTE_TANGENT)) {
        vec3 tangent =
            w * v0.tangent + u * v1.tangent + v * v2.tangent;
        tangent = tangent - dot(tangent, object_shading_normal) *
                                object_shading_normal;
        if (!tangent.near_zero()) {
            tangent = unit_vector(tangent);
            const double tangent_sign =
                w * v0.tangent_sign + u * v1.tangent_sign +
                        v * v2.tangent_sign <
                    0.0
                    ? -1.0
                    : 1.0;
            object_dpdu = tangent;
            object_dpdv =
                tangent_sign * cross(object_shading_normal, tangent);
        }
    } else if (has_mesh_attribute(triangle.attributes, MESH_ATTRIBUTE_UV0)) {
        const vec2 duv1 = v1.uv0 - v0.uv0;
        const vec2 duv2 = v2.uv0 - v0.uv0;
        const double determinant =
            duv1.x() * duv2.y() - duv1.y() * duv2.x();
        if (std::abs(determinant) > 1e-10) {
            const double inverse = 1.0 / determinant;
            object_dpdu =
                (duv2.y() * object_edge1 - duv1.y() * object_edge2) *
                inverse;
            object_dpdv =
                (-duv2.x() * object_edge1 + duv1.x() * object_edge2) *
                inverse;
        }
    }

    record.t = intersection.t;
    record.p = world_ray.at(intersection.t);
    record.front_face = dot(world_ray.direction(), outward_normal) < 0.0;
    record.geometric_normal =
        record.front_face ? outward_normal : -outward_normal;
    record.normal = record.front_face ? shading_normal : -shading_normal;
    record.dpdu = m_object_to_world.vector_to_world(object_dpdu);
    record.dpdv = m_object_to_world.vector_to_world(object_dpdv);
    if (has_mesh_attribute(triangle.attributes, MESH_ATTRIBUTE_UV0)) {
        record.u = w * v0.uv0.x() + u * v1.uv0.x() + v * v2.uv0.x();
        record.v = w * v0.uv0.y() + u * v1.uv0.y() + v * v2.uv0.y();
    } else {
        record.u = u;
        record.v = v;
    }
    record.mat_ptr = m_materials[triangle.material_slot].get();
    record.primitive_id = static_cast<int>(intersection.triangle_index);
    record.material_id = static_cast<int>(triangle.material_slot);
}

std::vector<TriangleSurface>
MeshInstance::light_triangles(int material_slot) const {
    std::vector<TriangleSurface> result;
    result.reserve(m_asset->triangle_count());
    for (std::uint32_t index = 0; index < m_asset->triangle_count(); ++index) {
        const MeshTriangle &triangle = m_asset->triangle(index);
        if (material_slot >= 0 &&
            triangle.material_slot != static_cast<std::uint32_t>(material_slot)) {
            continue;
        }
        TriangleSurface surface = m_asset->triangle_surface(index);
        surface.v0 = m_object_to_world.point_to_world(surface.v0);
        surface.v1 = m_object_to_world.point_to_world(surface.v1);
        surface.v2 = m_object_to_world.point_to_world(surface.v2);
        result.push_back(surface);
    }
    return result;
}

const std::shared_ptr<const MeshAsset> &MeshInstance::asset() const {
    return m_asset;
}

const Transform &MeshInstance::object_to_world() const {
    return m_object_to_world;
}

const std::vector<MaterialHandle> &MeshInstance::materials() const {
    return m_materials;
}

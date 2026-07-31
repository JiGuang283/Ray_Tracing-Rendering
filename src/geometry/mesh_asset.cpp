#include "mesh_asset.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace {

aabb triangle_bounds(const MeshVertex &v0, const MeshVertex &v1,
                     const MeshVertex &v2) {
    constexpr double padding = 1e-4;
    point3 minimum;
    point3 maximum;
    for (int axis = 0; axis < 3; ++axis) {
        minimum[axis] = std::min({v0.position[axis], v1.position[axis],
                                  v2.position[axis]}) -
                        padding;
        maximum[axis] = std::max({v0.position[axis], v1.position[axis],
                                  v2.position[axis]}) +
                        padding;
    }
    return aabb(minimum, maximum);
}

} // namespace

MeshAsset::MeshAsset(std::vector<MeshVertex> vertices,
                     std::vector<MeshTriangle> triangles,
                     std::vector<MeshPrimitive> primitives, bool build_bvh)
    : m_vertices(std::move(vertices)), m_triangles(std::move(triangles)),
      m_primitives(std::move(primitives)) {
    if (m_triangles.empty()) {
        throw std::invalid_argument("MeshAsset requires at least one triangle.");
    }

    std::vector<BuildTriangle> build_data;
    build_data.reserve(m_triangles.size());
    for (std::size_t triangle_index = 0;
         triangle_index < m_triangles.size(); ++triangle_index) {
        const MeshTriangle &triangle = m_triangles[triangle_index];
        for (std::uint32_t vertex_index : triangle.vertices) {
            if (vertex_index >= m_vertices.size()) {
                throw std::invalid_argument(
                    "MeshAsset triangle contains an invalid vertex index.");
            }
        }
        const MeshVertex &v0 = m_vertices[triangle.vertices[0]];
        const MeshVertex &v1 = m_vertices[triangle.vertices[1]];
        const MeshVertex &v2 = m_vertices[triangle.vertices[2]];
        const vec3 cross_product =
            cross(v1.position - v0.position, v2.position - v0.position);
        if (cross_product.length_squared() <= 1e-24) {
            throw std::invalid_argument(
                "MeshAsset contains a degenerate triangle.");
        }
        BuildTriangle build_triangle;
        build_triangle.bounds = triangle_bounds(v0, v1, v2);
        build_triangle.centroid =
            (v0.position + v1.position + v2.position) / 3.0;
        build_data.push_back(build_triangle);
        m_material_slot_count =
            std::max(m_material_slot_count,
                     static_cast<std::size_t>(triangle.material_slot) + 1);
    }

    if (m_primitives.empty()) {
        m_primitives.push_back(
            {"default", 0, static_cast<std::uint32_t>(m_triangles.size()), 0});
    }
    for (const MeshPrimitive &primitive : m_primitives) {
        const std::size_t end =
            static_cast<std::size_t>(primitive.first_triangle) +
            primitive.triangle_count;
        if (end > m_triangles.size()) {
            throw std::invalid_argument(
                "MeshAsset primitive range exceeds triangle data.");
        }
        m_material_slot_count =
            std::max(m_material_slot_count,
                     static_cast<std::size_t>(primitive.material_slot) + 1);
    }

    m_triangle_order.resize(m_triangles.size());
    std::iota(m_triangle_order.begin(), m_triangle_order.end(), 0u);
    m_bounds = build_data.front().bounds;
    for (std::size_t index = 1; index < build_data.size(); ++index) {
        m_bounds = surrounding_box(m_bounds, build_data[index].bounds);
    }
    if (build_bvh) {
        m_nodes.reserve(m_triangles.size() * 2);
        build_node(0, static_cast<std::uint32_t>(m_triangles.size()),
                   build_data);
    }
}

std::uint32_t
MeshAsset::build_node(std::uint32_t begin, std::uint32_t end,
                      const std::vector<BuildTriangle> &build_data) {
    const std::uint32_t node_index =
        static_cast<std::uint32_t>(m_nodes.size());
    m_nodes.emplace_back();

    aabb bounds = build_data[m_triangle_order[begin]].bounds;
    point3 centroid_min = build_data[m_triangle_order[begin]].centroid;
    point3 centroid_max = centroid_min;
    for (std::uint32_t index = begin + 1; index < end; ++index) {
        const BuildTriangle &triangle = build_data[m_triangle_order[index]];
        bounds = surrounding_box(bounds, triangle.bounds);
        for (int axis = 0; axis < 3; ++axis) {
            centroid_min[axis] =
                std::min(centroid_min[axis], triangle.centroid[axis]);
            centroid_max[axis] =
                std::max(centroid_max[axis], triangle.centroid[axis]);
        }
    }
    m_nodes[node_index].bounds = bounds;

    const std::uint32_t count = end - begin;
    if (count <= kLeafSize) {
        m_nodes[node_index].first_triangle = begin;
        m_nodes[node_index].triangle_count =
            static_cast<std::uint16_t>(count);
        return node_index;
    }

    const vec3 extent = centroid_max - centroid_min;
    int axis = 0;
    if (extent.y() > extent.x() && extent.y() > extent.z()) {
        axis = 1;
    } else if (extent.z() > extent.x()) {
        axis = 2;
    }
    const std::uint32_t middle = begin + count / 2;
    std::sort(
        m_triangle_order.begin() + begin, m_triangle_order.begin() + end,
        [&](std::uint32_t left, std::uint32_t right) {
            return build_data[left].centroid[axis] <
                   build_data[right].centroid[axis];
        });

    build_node(begin, middle, build_data);
    const std::uint32_t right = build_node(middle, end, build_data);
    m_nodes[node_index].second_child_offset = right;
    return node_index;
}

bool MeshAsset::intersect_triangle(std::uint32_t triangle_index,
                                   const ray &object_ray, double t_min,
                                   double t_max,
                                   MeshIntersection &intersection) const {
    const MeshTriangle &triangle = m_triangles[triangle_index];
    const point3 &v0 = m_vertices[triangle.vertices[0]].position;
    const vec3 edge1 = m_vertices[triangle.vertices[1]].position - v0;
    const vec3 edge2 = m_vertices[triangle.vertices[2]].position - v0;
    const vec3 pvec = cross(object_ray.direction(), edge2);
    const double determinant = dot(edge1, pvec);
    if (std::abs(determinant) < 1e-8) {
        return false;
    }

    const double inverse_determinant = 1.0 / determinant;
    const vec3 tvec = object_ray.origin() - v0;
    const double u = dot(tvec, pvec) * inverse_determinant;
    if (u < 0.0 || u > 1.0) {
        return false;
    }
    const vec3 qvec = cross(tvec, edge1);
    const double v =
        dot(object_ray.direction(), qvec) * inverse_determinant;
    if (v < 0.0 || u + v > 1.0) {
        return false;
    }
    const double t = dot(edge2, qvec) * inverse_determinant;
    if (t < t_min || t > t_max) {
        return false;
    }

    intersection.t = t;
    intersection.barycentric_u = u;
    intersection.barycentric_v = v;
    intersection.triangle_index = triangle_index;
    return true;
}

bool MeshAsset::intersect(const ray &object_ray, double t_min, double t_max,
                          MeshIntersection &intersection) const {
    bool found = false;
    double closest = t_max;
    MeshIntersection candidate;

    auto visit_range = [&](std::uint32_t first, std::uint32_t count) {
        for (std::uint32_t index = first; index < first + count; ++index) {
            const std::uint32_t triangle_index = m_triangle_order[index];
            if (intersect_triangle(triangle_index, object_ray, t_min, closest,
                                   candidate)) {
                closest = candidate.t;
                intersection = candidate;
                found = true;
            }
        }
    };

    if (m_nodes.empty()) {
        visit_range(0, static_cast<std::uint32_t>(m_triangle_order.size()));
        return found;
    }

    std::array<std::uint32_t, kTraversalStackSize> stack{};
    std::size_t stack_size = 0;
    std::uint32_t current = 0;
    while (true) {
        const MeshBVHNode &node = m_nodes[current];
        if (node.bounds.hit(object_ray, t_min, closest)) {
            if (node.is_leaf()) {
                visit_range(node.first_triangle, node.triangle_count);
                if (stack_size == 0) {
                    break;
                }
                current = stack[--stack_size];
            } else {
                if (stack_size == stack.size()) {
                    throw std::runtime_error(
                        "Mesh BVH traversal stack overflow.");
                }
                stack[stack_size++] = node.second_child_offset;
                current += 1;
            }
        } else {
            if (stack_size == 0) {
                break;
            }
            current = stack[--stack_size];
        }
    }
    return found;
}

const MeshVertex &MeshAsset::vertex(std::uint32_t index) const {
    return m_vertices.at(index);
}

const MeshTriangle &MeshAsset::triangle(std::uint32_t index) const {
    return m_triangles.at(index);
}

const std::vector<MeshPrimitive> &MeshAsset::primitives() const {
    return m_primitives;
}

const aabb &MeshAsset::bounds() const {
    return m_bounds;
}

std::size_t MeshAsset::vertex_count() const {
    return m_vertices.size();
}

std::size_t MeshAsset::triangle_count() const {
    return m_triangles.size();
}

std::size_t MeshAsset::node_count() const {
    return m_nodes.size();
}

std::size_t MeshAsset::material_slot_count() const {
    return m_material_slot_count;
}

TriangleSurface MeshAsset::triangle_surface(std::uint32_t index) const {
    const MeshTriangle &mesh_triangle = triangle(index);
    return {vertex(mesh_triangle.vertices[0]).position,
            vertex(mesh_triangle.vertices[1]).position,
            vertex(mesh_triangle.vertices[2]).position};
}

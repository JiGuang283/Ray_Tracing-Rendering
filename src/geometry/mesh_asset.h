#ifndef MESH_ASSET_H
#define MESH_ASSET_H

#include "aabb.h"
#include "triangle_surface.h"
#include "vec3.h"

#include <cstdint>
#include <string>
#include <vector>

enum MeshAttributeFlags : std::uint8_t {
    MESH_ATTRIBUTE_NONE = 0,
    MESH_ATTRIBUTE_NORMAL = 1 << 0,
    MESH_ATTRIBUTE_TANGENT = 1 << 1,
    MESH_ATTRIBUTE_UV0 = 1 << 2,
    MESH_ATTRIBUTE_COLOR0 = 1 << 3
};

inline bool has_mesh_attribute(std::uint8_t flags,
                               MeshAttributeFlags attribute) {
    return (flags & static_cast<std::uint8_t>(attribute)) != 0;
}

struct MeshVertex {
    point3 position{0, 0, 0};
    vec3 normal{0, 0, 1};
    vec3 tangent{1, 0, 0};
    vec2 uv0{0, 0};
    color color0{1, 1, 1};
    double color_alpha = 1.0;
    double tangent_sign = 1.0;
};

struct MeshTriangle {
    std::uint32_t vertices[3]{0, 0, 0};
    std::uint32_t primitive_index = 0;
    std::uint32_t material_slot = 0;
    std::uint8_t attributes = MESH_ATTRIBUTE_NONE;
};

struct MeshPrimitive {
    std::string name;
    std::uint32_t first_triangle = 0;
    std::uint32_t triangle_count = 0;
    std::uint32_t material_slot = 0;
};

struct MeshIntersection {
    double t = 0.0;
    double barycentric_u = 0.0;
    double barycentric_v = 0.0;
    std::uint32_t triangle_index = 0;
};

struct MeshBVHNode {
    aabb bounds;
    std::uint32_t second_child_offset = 0;
    std::uint32_t first_triangle = 0;
    std::uint16_t triangle_count = 0;

    bool is_leaf() const {
        return triangle_count != 0;
    }
};

class MeshAsset {
  public:
    MeshAsset(std::vector<MeshVertex> vertices,
              std::vector<MeshTriangle> triangles,
              std::vector<MeshPrimitive> primitives = {},
              bool build_bvh = true);

    bool intersect(const ray &object_ray, double t_min, double t_max,
                   MeshIntersection &intersection) const;

    const MeshVertex &vertex(std::uint32_t index) const;
    const MeshTriangle &triangle(std::uint32_t index) const;
    const std::vector<MeshPrimitive> &primitives() const;
    const aabb &bounds() const;
    std::size_t vertex_count() const;
    std::size_t triangle_count() const;
    std::size_t node_count() const;
    std::size_t material_slot_count() const;
    TriangleSurface triangle_surface(std::uint32_t index) const;

  private:
    struct BuildTriangle {
        aabb bounds;
        point3 centroid;
    };

    static constexpr std::uint32_t kLeafSize = 4;
    static constexpr std::size_t kTraversalStackSize = 128;

    std::uint32_t build_node(std::uint32_t begin, std::uint32_t end,
                             const std::vector<BuildTriangle> &build_data);
    bool intersect_triangle(std::uint32_t triangle_index, const ray &ray,
                            double t_min, double t_max,
                            MeshIntersection &intersection) const;

    std::vector<MeshVertex> m_vertices;
    std::vector<MeshTriangle> m_triangles;
    std::vector<MeshPrimitive> m_primitives;
    std::vector<std::uint32_t> m_triangle_order;
    std::vector<MeshBVHNode> m_nodes;
    aabb m_bounds;
    std::size_t m_material_slot_count = 0;
};

#endif

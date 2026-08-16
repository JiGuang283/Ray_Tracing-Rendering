#ifndef SCENE_COMPILER_INTERNAL_H
#define SCENE_COMPILER_INTERNAL_H

#include "scene_compiler.h"

#include "asset_path.h"
#include "material_programs.h"
#include "model_asset.h"
#include "packed_bvh.h"
#include "packed_material.h"
#include "resource_compiler.h"
#include "scene_resource_context.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>




namespace scene_compiler_detail {

using namespace scene_loader_internal;


inline std::uint32_t checked_index(std::size_t value, const char *what) {
    if (value >= kInvalidPackedIndex) {
        throw std::overflow_error(std::string(what) +
                                  " exceeds the packed 32-bit limit");
    }
    return static_cast<std::uint32_t>(value);
}

inline Range32 checked_range(std::size_t offset, std::size_t count,
                      const char *what) {
    if (offset >= kInvalidPackedIndex || count >= kInvalidPackedIndex ||
        count > kInvalidPackedIndex - offset) {
        throw std::overflow_error(std::string(what) +
                                  " exceeds the packed 32-bit limit");
    }
    return {static_cast<std::uint32_t>(offset),
            static_cast<std::uint32_t>(count)};
}

inline float checked_float(double value, const char *what) {
    if (!std::isfinite(value) ||
        value > static_cast<double>(std::numeric_limits<float>::max()) ||
        value < -static_cast<double>(std::numeric_limits<float>::max())) {
        throw std::runtime_error(std::string(what) +
                                 " is not representable as float32");
    }
    return static_cast<float>(value);
}

inline Float2 pack_vec2(const vec2 &value) {
    return {checked_float(value.x(), "vec2.x"),
            checked_float(value.y(), "vec2.y")};
}

inline Float3 pack_vec3(const vec3 &value) {
    return {checked_float(value.x(), "vec3.x"),
            checked_float(value.y(), "vec3.y"),
            checked_float(value.z(), "vec3.z")};
}

inline Float4 pack_vec4(const vec3 &value, double w) {
    return {checked_float(value.x(), "vec3.x"),
            checked_float(value.y(), "vec3.y"),
            checked_float(value.z(), "vec3.z"),
            checked_float(w, "vec4.w")};
}

inline double packed_luminance(const Float4 &value) {
    const double result = 0.2126 * value.x + 0.7152 * value.y +
                          0.0722 * value.z;
    return std::isfinite(result) ? std::max(0.0, result) : 0.0;
}

inline Float3 finite_nonnegative(Float3 value) {
    float *components[3]{&value.x, &value.y, &value.z};
    for (float *component : components) {
        *component = std::isfinite(*component)
                         ? std::max(0.0f, *component)
                         : 0.0f;
    }
    return value;
}

inline Float3 transform_packed_point(const PackedTransform &transform,
                              Float3 point) {
    const float *m = transform.object_to_world;
    return {m[0] * point.x + m[1] * point.y + m[2] * point.z + m[3],
            m[4] * point.x + m[5] * point.y + m[6] * point.z + m[7],
            m[8] * point.x + m[9] * point.y + m[10] * point.z + m[11]};
}

inline Float3 packed_subtract(Float3 a, Float3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline Float3 packed_cross(Float3 a, Float3 b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

inline double packed_length(Float3 value) {
    return std::sqrt(static_cast<double>(value.x) * value.x +
                     static_cast<double>(value.y) * value.y +
                     static_cast<double>(value.z) * value.z);
}

inline Float3 packed_barycentric(Float3 a, Float3 b, Float3 c,
                          const double weights[3]) {
    return {static_cast<float>(weights[0] * a.x + weights[1] * b.x +
                               weights[2] * c.x),
            static_cast<float>(weights[0] * a.y + weights[1] * b.y +
                               weights[2] * c.y),
            static_cast<float>(weights[0] * a.z + weights[1] * b.z +
                               weights[2] * c.z)};
}

inline PackedTransform pack_transform(const Transform &transform) {
    PackedTransform packed;
    const Matrix4 &object_to_world = transform.object_to_world();
    const Matrix4 &world_to_object = transform.world_to_object();
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 4; ++column) {
            const std::size_t index = row * 4 + column;
            packed.object_to_world[index] = checked_float(
                object_to_world(row, column), "object-to-world transform");
            packed.world_to_object[index] = checked_float(
                world_to_object(row, column), "world-to-object transform");
        }
    }
    return packed;
}

inline PackedCamera pack_camera(const CameraConfig &camera, double time0,
                         double time1) {
    const double theta = degrees_to_radians(camera.vfov);
    const double half_height = std::tan(theta / 2.0);
    const double viewport_height = 2.0 * half_height;
    const double viewport_width = camera.aspect_ratio * viewport_height;
    const vec3 w = unit_vector(camera.lookfrom - camera.lookat);
    const vec3 u = unit_vector(cross(camera.vup, w));
    const vec3 v = cross(w, u);
    const vec3 horizontal = camera.focus_dist * viewport_width * u;
    const vec3 vertical = camera.focus_dist * viewport_height * v;

    PackedCamera packed;
    packed.origin = pack_vec3(camera.lookfrom);
    packed.lens_radius = checked_float(camera.aperture / 2.0, "lens radius");
    packed.horizontal = pack_vec3(horizontal);
    packed.vertical = pack_vec3(vertical);
    packed.lower_left_corner = pack_vec3(
        camera.lookfrom - horizontal / 2.0 - vertical / 2.0 -
        camera.focus_dist * w);
    packed.time0 = checked_float(time0, "camera time0");
    packed.time1 = checked_float(time1, "camera time1");
    return packed;
}

inline aabb sphere_bounds(const point3 &center, double radius) {
    const double extent = std::abs(radius);
    const vec3 r(extent, extent, extent);
    return aabb(center - r, center + r);
}

inline aabb moving_sphere_bounds(const MovingSphereObjectIR &sphere, double time0,
                          double time1) {
    auto center_at = [&](double time) {
        const double duration = sphere.time1 - sphere.time0;
        if (std::abs(duration) <= 1e-15) {
            return sphere.center0;
        }
        return sphere.center0 +
               ((time - sphere.time0) / duration) *
                   (sphere.center1 - sphere.center0);
    };
    return surrounding_box(sphere_bounds(center_at(time0), sphere.radius),
                           sphere_bounds(center_at(time1), sphere.radius));
}

inline bool transform_is_identity(const Transform &transform) {
    const Matrix4 &matrix = transform.object_to_world();
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t column = 0; column < 4; ++column) {
            const double expected = row == column ? 1.0 : 0.0;
            if (std::abs(matrix(row, column) - expected) > 1e-12) {
                return false;
            }
        }
    }
    return true;
}

inline aabb triangle_bounds(const MeshVertex &v0, const MeshVertex &v1,
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
    return {minimum, maximum};
}

struct GeneratedMesh {
    std::vector<MeshVertex> vertices;
    std::vector<MeshTriangle> triangles;
    std::vector<MaterialHandle> materials;
    std::unordered_map<const MaterialInstance *, std::uint32_t>
        material_slots;

    std::uint32_t material_slot(const MaterialHandle &material) {
        auto found = material_slots.find(material.get());
        if (found != material_slots.end()) {
            return found->second;
        }
        const std::uint32_t slot =
            checked_index(materials.size(), "generated material slot count");
        materials.push_back(material);
        material_slots.emplace(material.get(), slot);
        return slot;
    }

    void add_triangle(const std::array<point3, 3> &positions,
                      const std::array<vec3, 3> &normals,
                      const std::array<vec2, 3> &uvs, bool has_normals,
                      bool has_uvs, const MaterialHandle &material,
                      const Transform &object_to_world, bool flip,
                      std::uint32_t source_id,
                      bool reverse_emitter_normal = false) {
        std::array<std::size_t, 3> order{0, flip ? 2u : 1u,
                                        flip ? 1u : 2u};
        const std::uint32_t first_vertex =
            checked_index(vertices.size(), "generated vertex count");
        std::array<point3, 3> world_positions;
        for (std::size_t corner = 0; corner < 3; ++corner) {
            const std::size_t source = order[corner];
            MeshVertex vertex;
            vertex.position =
                object_to_world.point_to_world(positions[source]);
            world_positions[corner] = vertex.position;
            if (has_normals) {
                vertex.normal = unit_vector(
                    object_to_world.normal_to_world(normals[source]));
            }
            vertex.uv0 = uvs[source];
            vertices.push_back(vertex);
        }

        const vec3 geometric_normal = unit_vector(
            cross(world_positions[1] - world_positions[0],
                  world_positions[2] - world_positions[0]));
        if (has_normals) {
            for (std::uint32_t corner = 0; corner < 3; ++corner) {
                MeshVertex &vertex = vertices[first_vertex + corner];
                if (dot(vertex.normal, geometric_normal) < 0.0) {
                    vertex.normal = -vertex.normal;
                }
            }
        }

        MeshTriangle triangle;
        triangle.vertices[0] = first_vertex;
        triangle.vertices[1] = first_vertex + 1;
        triangle.vertices[2] = first_vertex + 2;
        triangle.material_slot = material_slot(material);
        triangle.primitive_index = source_id;
        if (has_normals) {
            triangle.attributes |= MESH_ATTRIBUTE_NORMAL;
        }
        if (has_uvs) {
            triangle.attributes |= MESH_ATTRIBUTE_UV0;
        }
        if (reverse_emitter_normal) {
            triangle.flags |= MESH_TRIANGLE_REVERSE_EMITTER_NORMAL;
        }
        triangles.push_back(triangle);
    }

    void add_quad(const point3 &origin, const vec3 &u, const vec3 &v,
                  bool reverse_winding, const MaterialHandle &material,
                  const Transform &transform, bool flip,
                  std::uint32_t source_id,
                  bool reverse_emitter_normal = false) {
        const point3 p00 = origin;
        const point3 p10 = origin + u;
        const point3 p11 = origin + u + v;
        const point3 p01 = origin + v;
        const std::array<vec3, 3> no_normals{};
        if (!reverse_winding) {
            add_triangle({p00, p10, p11}, no_normals,
                         {vec2(0, 0), vec2(1, 0), vec2(1, 1)}, false, true,
                         material, transform, flip, source_id,
                         reverse_emitter_normal);
            add_triangle({p00, p11, p01}, no_normals,
                         {vec2(0, 0), vec2(1, 1), vec2(0, 1)}, false, true,
                         material, transform, flip, source_id,
                         reverse_emitter_normal);
        } else {
            add_triangle({p00, p01, p11}, no_normals,
                         {vec2(0, 0), vec2(0, 1), vec2(1, 1)}, false, true,
                         material, transform, flip, source_id,
                         reverse_emitter_normal);
            add_triangle({p00, p11, p10}, no_normals,
                         {vec2(0, 0), vec2(1, 1), vec2(1, 0)}, false, true,
                         material, transform, flip, source_id,
                         reverse_emitter_normal);
        }
    }
};

struct AggregateBuild {
    GeneratedMesh generated;
    std::vector<std::uint32_t> instances;
};


class SceneCompiler {
  public:
    explicit SceneCompiler(const SceneIR &ir);
CompiledScene compile();
private:
MaterialHandle material(const std::string &name,
                            const std::string &context);
std::uint32_t add_transform(const Transform &transform);
Range32 add_material_bindings(
        const std::vector<MaterialHandle> &materials);
std::uint32_t add_instance(PackedGeometryType type,
                               std::uint32_t geometry_index,
                               const Transform &transform,
                               const std::vector<MaterialHandle> &materials,
                               bool flip, std::uint32_t source_object_id,
                               const aabb &object_bounds,
                               AggregateBuild &aggregate);
std::uint32_t pack_mesh(const MeshAsset &asset);
std::uint32_t pack_shared_mesh(
        const std::shared_ptr<const MeshAsset> &asset);
void flush_generated(AggregateBuild &aggregate);
void finalize_aggregate(std::uint32_t aggregate_id,
                            AggregateBuild &aggregate);
std::uint32_t create_boundary_aggregate(ObjectIRId boundary,
                                            const Transform &transform);
void compile_obj(const ObjObjectIR &object, const ObjectIRNode &node,
                     const Transform &parent, bool flip,
                     AggregateBuild &aggregate);
void compile_model(const ModelObjectIR &object,
                       const ObjectIRNode &node, const Transform &parent,
                       bool flip, AggregateBuild &aggregate);
void compile_model_node(const ModelAsset &asset, std::size_t node_index,
                            const Transform &parent,
                            const std::vector<MaterialHandle> &materials,
                            bool flip, AggregateBuild &aggregate);
std::uint32_t append_light(PackedLight light);
Float3 mesh_position(const PackedMesh &mesh,
                         std::uint32_t vertex) const;
void append_mesh_emitter(std::uint32_t instance_id,
                             std::uint32_t material_slot,
                             std::uint32_t material_id);
void append_sphere_emitter(std::uint32_t instance_id,
                               std::uint32_t material_id);
void append_environment_light(const EnvironmentLightIR &environment);
void compile_explicit_lights();
void compile_auto_emitters();
void build_light_selection_distribution();
void compile_lights();
void compile_primitive(ObjectIRId id, const ObjectIRNode &node,
                           const Transform &transform, bool flip,
                           AggregateBuild &aggregate);
void compile_object(ObjectIRId id, const Transform &transform, bool flip,
                        AggregateBuild &aggregate, bool auto_emitters);
std::uint32_t current_source_id() const;

    const SceneIR &m_ir;
    CompiledScene m_scene;
    SceneResourceContext m_context;
    PackedResourceCompiler m_resources;
    std::unordered_map<const MeshAsset *, std::uint32_t> m_meshes;
    std::vector<aabb> m_instance_bounds;
    std::vector<aabb> m_aggregate_bounds;
    std::unordered_set<ObjectIRId> m_active_objects;
    ObjectIRId m_current_source = kInvalidObjectIR;
};

} // namespace scene_compiler_detail

#endif

#include "scene_compiler.h"

#include "material_programs.h"
#include "model_asset.h"
#include "packed_bvh.h"
#include "packed_material.h"
#include "resource_compiler.h"
#include "scene_loader.h"
#include "scene_loader_internal.h"

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

using namespace scene_loader_internal;

namespace {

std::uint32_t checked_index(std::size_t value, const char *what) {
    if (value >= kInvalidPackedIndex) {
        throw std::overflow_error(std::string(what) +
                                  " exceeds the packed 32-bit limit");
    }
    return static_cast<std::uint32_t>(value);
}

Range32 checked_range(std::size_t offset, std::size_t count,
                      const char *what) {
    if (offset >= kInvalidPackedIndex || count >= kInvalidPackedIndex ||
        count > kInvalidPackedIndex - offset) {
        throw std::overflow_error(std::string(what) +
                                  " exceeds the packed 32-bit limit");
    }
    return {static_cast<std::uint32_t>(offset),
            static_cast<std::uint32_t>(count)};
}

float checked_float(double value, const char *what) {
    if (!std::isfinite(value) ||
        value > static_cast<double>(std::numeric_limits<float>::max()) ||
        value < -static_cast<double>(std::numeric_limits<float>::max())) {
        throw std::runtime_error(std::string(what) +
                                 " is not representable as float32");
    }
    return static_cast<float>(value);
}

Float2 pack_vec2(const vec2 &value) {
    return {checked_float(value.x(), "vec2.x"),
            checked_float(value.y(), "vec2.y")};
}

Float3 pack_vec3(const vec3 &value) {
    return {checked_float(value.x(), "vec3.x"),
            checked_float(value.y(), "vec3.y"),
            checked_float(value.z(), "vec3.z")};
}

Float4 pack_vec4(const vec3 &value, double w) {
    return {checked_float(value.x(), "vec3.x"),
            checked_float(value.y(), "vec3.y"),
            checked_float(value.z(), "vec3.z"),
            checked_float(w, "vec4.w")};
}

double packed_luminance(const Float4 &value) {
    const double result = 0.2126 * value.x + 0.7152 * value.y +
                          0.0722 * value.z;
    return std::isfinite(result) ? std::max(0.0, result) : 0.0;
}

Float3 finite_nonnegative(Float3 value) {
    float *components[3]{&value.x, &value.y, &value.z};
    for (float *component : components) {
        *component = std::isfinite(*component)
                         ? std::max(0.0f, *component)
                         : 0.0f;
    }
    return value;
}

Float3 transform_packed_point(const PackedTransform &transform,
                              Float3 point) {
    const float *m = transform.object_to_world;
    return {m[0] * point.x + m[1] * point.y + m[2] * point.z + m[3],
            m[4] * point.x + m[5] * point.y + m[6] * point.z + m[7],
            m[8] * point.x + m[9] * point.y + m[10] * point.z + m[11]};
}

Float3 packed_subtract(Float3 a, Float3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Float3 packed_cross(Float3 a, Float3 b) {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

double packed_length(Float3 value) {
    return std::sqrt(static_cast<double>(value.x) * value.x +
                     static_cast<double>(value.y) * value.y +
                     static_cast<double>(value.z) * value.z);
}

Float3 packed_barycentric(Float3 a, Float3 b, Float3 c,
                          const double weights[3]) {
    return {static_cast<float>(weights[0] * a.x + weights[1] * b.x +
                               weights[2] * c.x),
            static_cast<float>(weights[0] * a.y + weights[1] * b.y +
                               weights[2] * c.y),
            static_cast<float>(weights[0] * a.z + weights[1] * b.z +
                               weights[2] * c.z)};
}

PackedTransform pack_transform(const Transform &transform) {
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

PackedCamera pack_camera(const CameraConfig &camera, double time0,
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

aabb sphere_bounds(const point3 &center, double radius) {
    const double extent = std::abs(radius);
    const vec3 r(extent, extent, extent);
    return aabb(center - r, center + r);
}

aabb moving_sphere_bounds(const MovingSphereObjectIR &sphere, double time0,
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
                      std::uint32_t source_id) {
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
        triangles.push_back(triangle);
    }

    void add_quad(const point3 &origin, const vec3 &u, const vec3 &v,
                  bool reverse_winding, const MaterialHandle &material,
                  const Transform &transform, bool flip,
                  std::uint32_t source_id) {
        const point3 p00 = origin;
        const point3 p10 = origin + u;
        const point3 p11 = origin + u + v;
        const point3 p01 = origin + v;
        const std::array<vec3, 3> no_normals{};
        if (!reverse_winding) {
            add_triangle({p00, p10, p11}, no_normals,
                         {vec2(0, 0), vec2(1, 0), vec2(1, 1)}, false, true,
                         material, transform, flip, source_id);
            add_triangle({p00, p11, p01}, no_normals,
                         {vec2(0, 0), vec2(1, 1), vec2(0, 1)}, false, true,
                         material, transform, flip, source_id);
        } else {
            add_triangle({p00, p01, p11}, no_normals,
                         {vec2(0, 0), vec2(0, 1), vec2(1, 1)}, false, true,
                         material, transform, flip, source_id);
            add_triangle({p00, p11, p10}, no_normals,
                         {vec2(0, 0), vec2(1, 1), vec2(1, 0)}, false, true,
                         material, transform, flip, source_id);
        }
    }
};

struct AggregateBuild {
    GeneratedMesh generated;
    std::vector<std::uint32_t> instances;
};

class SceneCompiler {
  public:
    explicit SceneCompiler(const SceneIR &ir)
        : m_ir(ir), m_resources(m_scene) {
        m_context.source_path = ir.source_path;
        m_context.scene_ir = &ir;
        m_scene.camera = pack_camera(ir.camera, ir.time0, ir.time1);
        m_scene.background = pack_vec3(ir.preset.background);
        m_scene.scene_time0 = checked_float(ir.time0, "scene time0");
        m_scene.scene_time1 = checked_float(ir.time1, "scene time1");
    }

    CompiledScene compile() {
        for (const MaterialIR &material : m_ir.materials) {
            m_context.materials[material.name] =
                build_material(material, m_context);
        }
        for (const auto &entry : m_context.materials) {
            m_resources.compile_material(entry.second);
        }

        m_scene.aggregates.emplace_back();
        m_aggregate_bounds.emplace_back();
        AggregateBuild world;
        for (ObjectIRId object : m_ir.objects) {
            compile_object(object, Transform(), false, world,
                           m_ir.auto_emitters);
        }
        finalize_aggregate(0, world);
        compile_lights();
        return std::move(m_scene);
    }

  private:
    MaterialHandle material(const std::string &name,
                            const std::string &context) {
        return lookup_material(m_context, name, context);
    }

    std::uint32_t add_transform(const Transform &transform) {
        const std::uint32_t id =
            checked_index(m_scene.transforms.size(), "transform count");
        m_scene.transforms.push_back(pack_transform(transform));
        return id;
    }

    Range32 add_material_bindings(
        const std::vector<MaterialHandle> &materials) {
        const std::size_t offset = m_scene.material_bindings.size();
        for (const MaterialHandle &material_handle : materials) {
            m_scene.material_bindings.push_back(
                m_resources.compile_material(material_handle).value);
        }
        return checked_range(offset, materials.size(), "material bindings");
    }

    std::uint32_t add_instance(PackedGeometryType type,
                               std::uint32_t geometry_index,
                               const Transform &transform,
                               const std::vector<MaterialHandle> &materials,
                               bool flip, std::uint32_t source_object_id,
                               const aabb &object_bounds,
                               AggregateBuild &aggregate) {
        const aabb world_bounds = transform.bounds_to_world(object_bounds);
        const PackedBVHNode packed_bounds = pack_packed_bounds(world_bounds);
        PackedInstance instance;
        instance.geometry_type = type;
        instance.geometry_index = geometry_index;
        instance.transform_id = add_transform(transform);
        instance.flags = flip ? PACKED_INSTANCE_FLIP_FACE
                              : PACKED_INSTANCE_NONE;
        instance.material_bindings = add_material_bindings(materials);
        instance.source_object_id = source_object_id;
        instance.bounds_min = packed_bounds.bounds_min;
        instance.bounds_max = packed_bounds.bounds_max;

        const std::uint32_t id =
            checked_index(m_scene.instances.size(), "instance count");
        m_scene.instances.push_back(instance);
        m_instance_bounds.push_back(world_bounds);
        aggregate.instances.push_back(id);
        return id;
    }

    std::uint32_t pack_mesh(const MeshAsset &asset) {
        const std::size_t vertex_offset = m_scene.positions.size();
        for (std::size_t index = 0; index < asset.vertex_count(); ++index) {
            const MeshVertex &vertex =
                asset.vertex(static_cast<std::uint32_t>(index));
            m_scene.positions.push_back(pack_vec4(vertex.position, 1.0));
            m_scene.normals.push_back(pack_vec4(vertex.normal, 0.0));
            m_scene.tangents.push_back(
                pack_vec4(vertex.tangent, vertex.tangent_sign));
            m_scene.uv0.push_back(pack_vec2(vertex.uv0));
            m_scene.vertex_colors.push_back(
                pack_vec4(vertex.color0, vertex.color_alpha));
        }

        std::vector<PackedBVHPrimitive> build_primitives;
        build_primitives.reserve(asset.triangle_count());
        for (std::size_t index = 0; index < asset.triangle_count(); ++index) {
            const MeshTriangle &triangle =
                asset.triangle(static_cast<std::uint32_t>(index));
            build_primitives.push_back(
                {triangle_bounds(asset.vertex(triangle.vertices[0]),
                                 asset.vertex(triangle.vertices[1]),
                                 asset.vertex(triangle.vertices[2])),
                 static_cast<std::uint32_t>(index)});
        }
        const PackedBVHBuildResult build =
            build_packed_bvh(std::move(build_primitives));
        const std::size_t triangle_offset = m_scene.triangles.size();
        for (std::uint32_t source_index : build.ordered_payloads) {
            const MeshTriangle &source = asset.triangle(source_index);
            PackedTriangle triangle;
            triangle.vertex0 = source.vertices[0];
            triangle.vertex1 = source.vertices[1];
            triangle.vertex2 = source.vertices[2];
            triangle.material_slot = source.material_slot;
            triangle.primitive_id = source.primitive_index;
            if (has_mesh_attribute(source.attributes,
                                   MESH_ATTRIBUTE_NORMAL)) {
                triangle.flags |= PACKED_TRIANGLE_HAS_NORMALS;
            }
            if (has_mesh_attribute(source.attributes, MESH_ATTRIBUTE_UV0)) {
                triangle.flags |= PACKED_TRIANGLE_HAS_UV;
            }
            if (has_mesh_attribute(source.attributes,
                                   MESH_ATTRIBUTE_COLOR0)) {
                triangle.flags |= PACKED_TRIANGLE_HAS_COLOR;
            }
            if (has_mesh_attribute(source.attributes,
                                   MESH_ATTRIBUTE_TANGENT)) {
                triangle.flags |= PACKED_TRIANGLE_HAS_TANGENT;
            }
            m_scene.triangles.push_back(triangle);
        }
        const Range32 bvh_nodes = append_packed_bvh(
            m_scene, build,
            checked_index(triangle_offset, "triangle buffer offset"));

        const PackedBVHNode bounds = pack_packed_bounds(asset.bounds());
        PackedMesh mesh;
        mesh.vertices =
            checked_range(vertex_offset, asset.vertex_count(), "mesh vertices");
        mesh.triangles = checked_range(triangle_offset,
                                       asset.triangle_count(),
                                       "mesh triangles");
        mesh.bvh_nodes = bvh_nodes;
        mesh.material_slot_count = checked_index(
            asset.material_slot_count(), "mesh material slot count");
        mesh.bounds_min = bounds.bounds_min;
        mesh.bounds_max = bounds.bounds_max;
        const std::uint32_t id =
            checked_index(m_scene.meshes.size(), "mesh count");
        m_scene.meshes.push_back(mesh);
        return id;
    }

    std::uint32_t pack_shared_mesh(
        const std::shared_ptr<const MeshAsset> &asset) {
        auto found = m_meshes.find(asset.get());
        if (found != m_meshes.end()) {
            return found->second;
        }
        const std::uint32_t id = pack_mesh(*asset);
        m_meshes.emplace(asset.get(), id);
        return id;
    }

    void flush_generated(AggregateBuild &aggregate) {
        if (aggregate.generated.triangles.empty()) {
            return;
        }
        auto asset = std::make_shared<MeshAsset>(
            std::move(aggregate.generated.vertices),
            std::move(aggregate.generated.triangles),
            std::vector<MeshPrimitive>{}, false);
        const std::uint32_t mesh_id = pack_mesh(*asset);
        add_instance(PackedGeometryType::Mesh, mesh_id, Transform(),
                     aggregate.generated.materials, false,
                     kInvalidPackedIndex, asset->bounds(), aggregate);
    }

    void finalize_aggregate(std::uint32_t aggregate_id,
                            AggregateBuild &aggregate) {
        flush_generated(aggregate);
        std::vector<PackedBVHPrimitive> primitives;
        primitives.reserve(aggregate.instances.size());
        bool initialized = false;
        aabb aggregate_bounds;
        for (std::uint32_t instance : aggregate.instances) {
            const aabb &bounds = m_instance_bounds[instance];
            primitives.push_back({bounds, instance});
            aggregate_bounds = initialized
                                   ? surrounding_box(aggregate_bounds, bounds)
                                   : bounds;
            initialized = true;
        }

        PackedAggregate packed;
        const std::size_t instance_offset =
            m_scene.aggregate_instance_indices.size();
        if (!primitives.empty()) {
            const PackedBVHBuildResult build =
                build_packed_bvh(std::move(primitives));
            m_scene.aggregate_instance_indices.insert(
                m_scene.aggregate_instance_indices.end(),
                build.ordered_payloads.begin(), build.ordered_payloads.end());
            packed.instance_indices = checked_range(
                instance_offset, build.ordered_payloads.size(),
                "aggregate instance indices");
            packed.bvh_nodes = append_packed_bvh(
                m_scene, build,
                checked_index(instance_offset,
                              "aggregate instance buffer offset"));
            m_aggregate_bounds[aggregate_id] = aggregate_bounds;
        }
        m_scene.aggregates[aggregate_id] = packed;
    }

    std::uint32_t create_boundary_aggregate(ObjectIRId boundary,
                                            const Transform &transform) {
        const std::uint32_t id =
            checked_index(m_scene.aggregates.size(), "aggregate count");
        m_scene.aggregates.emplace_back();
        m_aggregate_bounds.emplace_back();
        AggregateBuild aggregate;
        compile_object(boundary, transform, false, aggregate, false);
        finalize_aggregate(id, aggregate);
        if (aggregate.instances.empty()) {
            throw std::runtime_error(
                "constant medium boundary compiled to an empty aggregate");
        }
        return id;
    }

    void compile_obj(const ObjObjectIR &object, const ObjectIRNode &node,
                     const Transform &parent, bool flip,
                     AggregateBuild &aggregate) {
        const MaterialHandle object_material =
            material(object.material, node.context);
        const std::string path = resolve_asset_path(m_context, object.path);
        std::string error;
        const auto asset = m_context.resources.load_obj(
            path, object.build_bvh, object.use_vertex_normals, error);
        if (!asset) {
            throw std::runtime_error("Scene compile error: failed to load OBJ '" +
                                     path + "': " + error);
        }

        Transform local = Transform::rotate_y(object.rotation_y) *
                          Transform::translate(object.local_translation) *
                          Transform::scale(object.scale);
        vec3 position = object.position;
        if (object.auto_lift_to_ground) {
            const aabb transformed = local.bounds_to_world(asset->bounds());
            position[1] -= transformed.min().y();
            local = Transform::translate(position) * local;
        } else if (object.has_position) {
            local = Transform::translate(position) * local;
        }
        add_instance(PackedGeometryType::Mesh, pack_shared_mesh(asset),
                     parent * local, {object_material}, flip,
                     current_source_id(), asset->bounds(), aggregate);
    }

    void compile_model(const ModelObjectIR &object,
                       const ObjectIRNode &node, const Transform &parent,
                       bool flip, AggregateBuild &aggregate) {
        const std::string path = resolve_asset_path(m_context, object.path);
        ModelImportOptions options;
        std::string error;
        const auto asset =
            m_context.resources.load_model(path, options, error);
        if (!asset) {
            throw std::runtime_error(
                "Scene compile error: failed to load model '" + path +
                "': " + error);
        }

        std::vector<MaterialHandle> materials = asset->materials();
        for (const auto &override_entry : object.material_overrides) {
            bool matched = false;
            for (std::size_t slot = 0;
                 slot < asset->material_names().size(); ++slot) {
                if (asset->material_names()[slot] == override_entry.first) {
                    materials[slot] = material(override_entry.second,
                                               node.context);
                    matched = true;
                }
            }
            if (!matched) {
                throw std::runtime_error(
                    "Scene compile error: unknown model material override '" +
                    override_entry.first + "'");
            }
        }
        for (const MaterialHandle &entry : materials) {
            m_resources.compile_material(entry);
        }

        const int scene_index = asset->resolve_scene_index(object.scene_index);
        const ModelScene &model_scene =
            asset->scenes()[static_cast<std::size_t>(scene_index)];
        for (std::size_t root : model_scene.roots) {
            compile_model_node(*asset, root, parent * object.transform,
                               materials, flip, aggregate);
        }
    }

    void compile_model_node(const ModelAsset &asset, std::size_t node_index,
                            const Transform &parent,
                            const std::vector<MaterialHandle> &materials,
                            bool flip, AggregateBuild &aggregate) {
        if (node_index >= asset.nodes().size()) {
            throw std::runtime_error(
                "Scene compile error: model node index is invalid");
        }
        const ModelNode &node = asset.nodes()[node_index];
        const Transform node_to_world = parent * node.local_transform;
        if (node.mesh_index >= 0) {
            const ModelMesh &mesh =
                asset.meshes()[static_cast<std::size_t>(node.mesh_index)];
            add_instance(PackedGeometryType::Mesh,
                         pack_shared_mesh(mesh.geometry), node_to_world,
                         materials, flip, current_source_id(),
                         mesh.geometry->bounds(), aggregate);
        }
        for (std::size_t child : node.children) {
            compile_model_node(asset, child, node_to_world, materials, flip,
                               aggregate);
        }
    }

    std::uint32_t append_light(PackedLight light) {
        const std::uint32_t id =
            checked_index(m_scene.lights.size(), "light count");
        m_scene.lights.push_back(light);
        if ((light.flags & PACKED_LIGHT_DELTA) != 0) {
            m_scene.delta_light_indices.push_back(id);
        } else {
            m_scene.non_delta_light_indices.push_back(id);
        }
        return id;
    }

    Float3 mesh_position(const PackedMesh &mesh,
                         std::uint32_t vertex) const {
        const Float4 &value = m_scene.positions[mesh.vertices.offset + vertex];
        return {value.x, value.y, value.z};
    }

    void append_mesh_emitter(std::uint32_t instance_id,
                             std::uint32_t material_slot,
                             std::uint32_t material_id) {
        struct EmitterTriangle {
            std::uint32_t index = 0;
            double area = 0.0;
            Float3 emission;
        };

        const PackedInstance &instance = m_scene.instances[instance_id];
        const PackedMesh &mesh = m_scene.meshes[instance.geometry_index];
        const PackedTransform &transform =
            m_scene.transforms[instance.transform_id];
        constexpr double sample_points[4][3] = {
            {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0},
            {0.6, 0.2, 0.2},
            {0.2, 0.6, 0.2},
            {0.2, 0.2, 0.6}};
        std::vector<EmitterTriangle> entries;
        double total_area = 0.0;
        Float3 integrated{};
        for (std::uint32_t local = 0; local < mesh.triangles.count; ++local) {
            const std::uint32_t triangle_id = mesh.triangles.offset + local;
            const PackedTriangle &triangle = m_scene.triangles[triangle_id];
            if (triangle.material_slot != material_slot) {
                continue;
            }
            const Float3 object0 = mesh_position(mesh, triangle.vertex0);
            const Float3 object1 = mesh_position(mesh, triangle.vertex1);
            const Float3 object2 = mesh_position(mesh, triangle.vertex2);
            const Float3 world0 = transform_packed_point(transform, object0);
            const Float3 world1 = transform_packed_point(transform, object1);
            const Float3 world2 = transform_packed_point(transform, object2);
            const double area =
                0.5 * packed_length(packed_cross(
                          packed_subtract(world1, world0),
                          packed_subtract(world2, world0)));
            if (!(area > 0.0) || !std::isfinite(area)) {
                continue;
            }

            const Float2 uv0 =
                m_scene.uv0[mesh.vertices.offset + triangle.vertex0];
            const Float2 uv1 =
                m_scene.uv0[mesh.vertices.offset + triangle.vertex1];
            const Float2 uv2 =
                m_scene.uv0[mesh.vertices.offset + triangle.vertex2];
            const Float4 color0 = m_scene.vertex_colors[
                mesh.vertices.offset + triangle.vertex0];
            const Float4 color1 = m_scene.vertex_colors[
                mesh.vertices.offset + triangle.vertex1];
            const Float4 color2 = m_scene.vertex_colors[
                mesh.vertices.offset + triangle.vertex2];
            Float3 estimate{};
            for (const auto &weights : sample_points) {
                PackedTextureEvalContext context;
                context.position =
                    packed_barycentric(world0, world1, world2, weights);
                if ((triangle.flags & PACKED_TRIANGLE_HAS_UV) != 0) {
                    context.uv0 = {
                        static_cast<float>(weights[0] * uv0.x +
                                           weights[1] * uv1.x +
                                           weights[2] * uv2.x),
                        static_cast<float>(weights[0] * uv0.y +
                                           weights[1] * uv1.y +
                                           weights[2] * uv2.y)};
                } else {
                    context.uv0 = {static_cast<float>(weights[1]),
                                   static_cast<float>(weights[2])};
                }
                if ((triangle.flags & PACKED_TRIANGLE_HAS_COLOR) != 0) {
                    context.vertex_color = {
                        static_cast<float>(weights[0] * color0.x +
                                           weights[1] * color1.x +
                                           weights[2] * color2.x),
                        static_cast<float>(weights[0] * color0.y +
                                           weights[1] * color1.y +
                                           weights[2] * color2.y),
                        static_cast<float>(weights[0] * color0.z +
                                           weights[1] * color1.z +
                                           weights[2] * color2.z),
                        static_cast<float>(weights[0] * color0.w +
                                           weights[1] * color1.w +
                                           weights[2] * color2.w)};
                }
                Float3 emission;
                if (!evaluate_packed_material_emission(
                        make_scene_view(m_scene), material_id, context,
                        emission)) {
                    throw std::runtime_error(
                        "failed to evaluate packed mesh emission");
                }
                estimate.x += emission.x;
                estimate.y += emission.y;
                estimate.z += emission.z;
            }
            estimate = finite_nonnegative(
                {estimate.x * 0.25f, estimate.y * 0.25f,
                 estimate.z * 0.25f});
            entries.push_back({triangle_id, area, estimate});
            total_area += area;
            integrated.x += static_cast<float>(area * estimate.x);
            integrated.y += static_cast<float>(area * estimate.y);
            integrated.z += static_cast<float>(area * estimate.z);
        }
        if (entries.empty() || !(total_area > 0.0)) {
            return;
        }

        double emission_weight_sum = 0.0;
        for (const EmitterTriangle &entry : entries) {
            emission_weight_sum +=
                entry.area * (0.2126 * entry.emission.x +
                              0.7152 * entry.emission.y +
                              0.0722 * entry.emission.z);
        }
        const std::size_t element_offset =
            m_scene.light_element_indices.size();
        const std::size_t distribution_offset =
            m_scene.light_distributions.size();
        std::vector<float> probabilities;
        probabilities.reserve(entries.size());
        for (const EmitterTriangle &entry : entries) {
            const double area_probability = entry.area / total_area;
            const double emission_probability =
                emission_weight_sum > 0.0
                    ? entry.area *
                          (0.2126 * entry.emission.x +
                           0.7152 * entry.emission.y +
                           0.0722 * entry.emission.z) /
                          emission_weight_sum
                    : area_probability;
            const float probability = static_cast<float>(
                0.95 * emission_probability + 0.05 * area_probability);
            probabilities.push_back(probability);
            m_scene.light_distributions.push_back(probability);
            m_scene.light_element_indices.push_back(entry.index);
        }
        float cumulative = 0.0f;
        for (float probability : probabilities) {
            cumulative += probability;
            m_scene.light_distributions.push_back(cumulative);
        }
        m_scene.light_distributions.back() = 1.0f;

        const PackedMaterial &material = m_scene.materials[material_id];
        const float side_factor =
            (material.flags & PACKED_MATERIAL_DOUBLE_SIDED) != 0 ? 2.0f
                                                                  : 1.0f;
        PackedLight light;
        light.type = entries.size() == 1
                         ? PackedLightType::TriangleEmitter
                         : PackedLightType::MeshEmitter;
        light.flags = PACKED_LIGHT_BSDF_HITTABLE;
        if (side_factor == 2.0f) {
            light.flags |= PACKED_LIGHT_DOUBLE_SIDED;
        }
        light.instance_id = instance_id;
        light.material_id = material_id;
        light.element_indices = checked_range(
            element_offset, entries.size(), "light triangle indices");
        light.distribution = checked_range(
            distribution_offset, entries.size() * 2,
            "light triangle distribution");
        light.data0 = {checked_float(total_area, "emitter area"),
                       static_cast<float>(entries.size()), 0.0f, 0.0f};
        light.radiance = {
            integrated.x / static_cast<float>(total_area),
            integrated.y / static_cast<float>(total_area),
            integrated.z / static_cast<float>(total_area), 0.0f};
        light.power = {side_factor * static_cast<float>(pi) * integrated.x,
                       side_factor * static_cast<float>(pi) * integrated.y,
                       side_factor * static_cast<float>(pi) * integrated.z,
                       0.0f};
        append_light(light);
    }

    void append_sphere_emitter(std::uint32_t instance_id,
                               std::uint32_t material_id) {
        const PackedInstance &instance = m_scene.instances[instance_id];
        const PackedSphere &sphere = m_scene.spheres[instance.geometry_index];
        const PackedTransform &transform =
            m_scene.transforms[instance.transform_id];
        constexpr Float3 normals[6] = {
            {1, 0, 0}, {-1, 0, 0}, {0, 1, 0},
            {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
        const float *m = transform.object_to_world;
        const double determinant =
            m[0] * (m[5] * m[10] - m[6] * m[9]) -
            m[1] * (m[4] * m[10] - m[6] * m[8]) +
            m[2] * (m[4] * m[9] - m[5] * m[8]);
        const float *inverse = transform.world_to_object;
        const double base_patch =
            4.0 * pi * sphere.radius * sphere.radius / 6.0;
        double total_area = 0.0;
        Float3 integrated{};
        for (Float3 normal : normals) {
            const Float3 inverse_transpose{
                inverse[0] * normal.x + inverse[4] * normal.y +
                    inverse[8] * normal.z,
                inverse[1] * normal.x + inverse[5] * normal.y +
                    inverse[9] * normal.z,
                inverse[2] * normal.x + inverse[6] * normal.y +
                    inverse[10] * normal.z};
            const double area = base_patch * std::abs(determinant) *
                                packed_length(inverse_transpose);
            PackedTextureEvalContext context;
            const Float3 object_position{
                sphere.center.x + sphere.radius * normal.x,
                sphere.center.y + sphere.radius * normal.y,
                sphere.center.z + sphere.radius * normal.z};
            context.position =
                transform_packed_point(transform, object_position);
            if ((sphere.flags & PACKED_SPHERE_FLIP_ORIENTATION) != 0) {
                normal = {-normal.x, -normal.y, -normal.z};
            }
            const double theta = std::acos(std::clamp(
                -static_cast<double>(normal.y), -1.0, 1.0));
            const double phi =
                std::atan2(-static_cast<double>(normal.z), normal.x) + pi;
            context.uv0 = {static_cast<float>(phi / (2.0 * pi)),
                           static_cast<float>(theta / pi)};
            Float3 emission;
            if (!evaluate_packed_material_emission(
                    make_scene_view(m_scene), material_id, context,
                    emission)) {
                throw std::runtime_error(
                    "failed to evaluate packed sphere emission");
            }
            emission = finite_nonnegative(emission);
            total_area += area;
            integrated.x += static_cast<float>(area * emission.x);
            integrated.y += static_cast<float>(area * emission.y);
            integrated.z += static_cast<float>(area * emission.z);
        }
        if (!(total_area > 0.0) || !std::isfinite(total_area)) {
            return;
        }
        const PackedMaterial &material = m_scene.materials[material_id];
        const float side_factor =
            (material.flags & PACKED_MATERIAL_DOUBLE_SIDED) != 0 ? 2.0f
                                                                  : 1.0f;
        PackedLight light;
        light.type = PackedLightType::SphereEmitter;
        light.flags = PACKED_LIGHT_BSDF_HITTABLE;
        if (side_factor == 2.0f) {
            light.flags |= PACKED_LIGHT_DOUBLE_SIDED;
        }
        light.instance_id = instance_id;
        light.material_id = material_id;
        light.data0 = {sphere.center.x, sphere.center.y, sphere.center.z,
                       sphere.radius};
        light.data1.x = checked_float(total_area, "sphere emitter area");
        light.radiance = {
            integrated.x / static_cast<float>(total_area),
            integrated.y / static_cast<float>(total_area),
            integrated.z / static_cast<float>(total_area), 0.0f};
        light.power = {side_factor * static_cast<float>(pi) * integrated.x,
                       side_factor * static_cast<float>(pi) * integrated.y,
                       side_factor * static_cast<float>(pi) * integrated.z,
                       0.0f};
        append_light(light);
    }

    void append_environment_light(const EnvironmentLightIR &environment) {
        const std::string path =
            resolve_asset_path(m_context, environment.path);
        const auto image = m_context.resources.load_image(path);
        const ImageId image_id = m_resources.compile_image(image);
        const int width = image->width();
        const int height = image->height();
        auto linear_component = [&](int x, int y, int channel) {
            double value = image->component(x, y, channel);
            if (!image->is_hdr()) {
                value = value <= 0.04045
                            ? value / 12.92
                            : std::pow((value + 0.055) / 1.055, 2.4);
            }
            return value;
        };
        std::vector<double> weights(
            static_cast<std::size_t>(width) * height);
        std::vector<double> row_sums(static_cast<std::size_t>(height), 0.0);
        double total_weight = 0.0;
        for (int y = 0; y < height; ++y) {
            const double sin_theta =
                std::sin(pi * (static_cast<double>(y) + 0.5) / height);
            for (int x = 0; x < width; ++x) {
                const double luminance =
                    0.2126 * linear_component(x, y, 0) +
                    0.7152 * linear_component(x, y, 1) +
                    0.0722 * linear_component(x, y, 2);
                const double weight = std::max(0.0, luminance * sin_theta);
                weights[static_cast<std::size_t>(y) * width + x] = weight;
                row_sums[static_cast<std::size_t>(y)] += weight;
                total_weight += weight;
            }
        }

        const std::size_t offset = m_scene.light_distributions.size();
        for (int y = 0; y < height; ++y) {
            const double row_sum = row_sums[static_cast<std::size_t>(y)];
            for (int x = 0; x < width; ++x) {
                const double value =
                    weights[static_cast<std::size_t>(y) * width + x];
                m_scene.light_distributions.push_back(
                    row_sum > 0.0 ? static_cast<float>(value / row_sum)
                                  : 0.0f);
            }
        }
        for (int y = 0; y < height; ++y) {
            float cumulative = 0.0f;
            m_scene.light_distributions.push_back(0.0f);
            const double row_sum = row_sums[static_cast<std::size_t>(y)];
            for (int x = 0; x < width; ++x) {
                const double value =
                    weights[static_cast<std::size_t>(y) * width + x];
                cumulative += row_sum > 0.0
                                  ? static_cast<float>(value / row_sum)
                                  : 0.0f;
                m_scene.light_distributions.push_back(cumulative);
            }
            if (row_sum > 0.0) {
                m_scene.light_distributions.back() = 1.0f;
            }
        }
        for (double row_sum : row_sums) {
            m_scene.light_distributions.push_back(
                total_weight > 0.0
                    ? static_cast<float>(row_sum / total_weight)
                    : 0.0f);
        }
        float cumulative = 0.0f;
        m_scene.light_distributions.push_back(0.0f);
        for (double row_sum : row_sums) {
            cumulative += total_weight > 0.0
                              ? static_cast<float>(row_sum / total_weight)
                              : 0.0f;
            m_scene.light_distributions.push_back(cumulative);
        }
        if (total_weight > 0.0) {
            m_scene.light_distributions.back() = 1.0f;
        }

        PackedLight light;
        light.type = PackedLightType::Environment;
        light.flags = PACKED_LIGHT_INFINITE | PACKED_LIGHT_BSDF_HITTABLE;
        if (width == height) {
            light.flags |= PACKED_LIGHT_ENVIRONMENT_PROBE;
        }
        if (!image->is_hdr()) {
            light.flags |= PACKED_LIGHT_ENVIRONMENT_SRGB;
        }
        light.image_id = image_id.value;
        light.distribution = checked_range(
            offset, m_scene.light_distributions.size() - offset,
            "environment distribution");
        light.data0 = {static_cast<float>(width),
                       static_cast<float>(height), 0.0f, 0.0f};
        const double total_power =
            total_weight * (2.0 * pi * pi) /
            (static_cast<double>(width) * height);
        light.power = {checked_float(total_power, "environment power"),
                       checked_float(total_power, "environment power"),
                       checked_float(total_power, "environment power"),
                       0.0f};
        append_light(light);
    }

    void compile_explicit_lights() {
        for (const LightIR &source : m_ir.lights) {
            std::visit(
                [&](const auto &light_ir) {
                    using T = std::decay_t<decltype(light_ir)>;
                    PackedLight light;
                    if constexpr (std::is_same_v<T, PointLightIR>) {
                        light.type = PackedLightType::Point;
                        light.flags = PACKED_LIGHT_DELTA;
                        light.data0 = pack_vec4(light_ir.position, 1.0);
                        light.radiance = pack_vec4(light_ir.intensity, 0.0);
                        light.power =
                            pack_vec4(4.0 * pi * light_ir.intensity, 0.0);
                        append_light(light);
                    } else if constexpr (std::is_same_v<T,
                                                        DirectionalLightIR>) {
                        if (light_ir.direction.near_zero()) {
                            throw std::runtime_error(
                                "directional light has a zero direction");
                        }
                        light.type = PackedLightType::Directional;
                        light.flags = PACKED_LIGHT_DELTA;
                        light.data0 =
                            pack_vec4(unit_vector(light_ir.direction), 0.0);
                        light.radiance = pack_vec4(light_ir.radiance, 0.0);
                        light.power = pack_vec4(light_ir.radiance, 0.0);
                        append_light(light);
                    } else if constexpr (std::is_same_v<T, SpotLightIR>) {
                        if (light_ir.direction.near_zero()) {
                            throw std::runtime_error(
                                "spot light has a zero direction");
                        }
                        light.type = PackedLightType::Spot;
                        light.flags = PACKED_LIGHT_DELTA;
                        light.data0 = pack_vec4(
                            light_ir.position,
                            std::cos(degrees_to_radians(light_ir.cutoff)));
                        light.data1 =
                            pack_vec4(unit_vector(light_ir.direction), 0.0);
                        light.radiance = pack_vec4(light_ir.intensity, 0.0);
                        light.power = pack_vec4(light_ir.intensity, 0.0);
                        append_light(light);
                    } else if constexpr (std::is_same_v<T, QuadLightIR>) {
                        const double area =
                            cross(light_ir.u, light_ir.v).length();
                        light.type = PackedLightType::Quad;
                        light.data0 = pack_vec4(light_ir.origin, 1.0);
                        light.data1 = pack_vec4(light_ir.u, 0.0);
                        light.data2 = pack_vec4(light_ir.v, 0.0);
                        light.radiance = pack_vec4(light_ir.intensity, 0.0);
                        light.power = pack_vec4(
                            pi * area * light_ir.intensity, 0.0);
                        append_light(light);
                    } else if constexpr (std::is_same_v<
                                             T, EnvironmentLightIR>) {
                        append_environment_light(light_ir);
                    }
                },
                source.data);
        }
    }

    void compile_auto_emitters() {
        if (!m_ir.auto_emitters || m_scene.aggregates.empty()) {
            return;
        }
        const PackedAggregate &world = m_scene.aggregates[0];
        for (std::uint32_t local = 0; local < world.instance_indices.count;
             ++local) {
            const std::uint32_t instance_id =
                m_scene.aggregate_instance_indices[
                    world.instance_indices.offset + local];
            const PackedInstance &instance = m_scene.instances[instance_id];
            if (instance.geometry_type == PackedGeometryType::Sphere) {
                const std::uint32_t material_id =
                    m_scene.material_bindings[
                        instance.material_bindings.offset];
                if ((m_scene.materials[material_id].flags &
                     PACKED_MATERIAL_EMISSIVE) != 0) {
                    append_sphere_emitter(instance_id, material_id);
                }
            } else if (instance.geometry_type == PackedGeometryType::Mesh) {
                const PackedMesh &mesh =
                    m_scene.meshes[instance.geometry_index];
                for (std::uint32_t slot = 0;
                     slot < mesh.material_slot_count; ++slot) {
                    const std::uint32_t material_id =
                        m_scene.material_bindings[
                            instance.material_bindings.offset + slot];
                    if ((m_scene.materials[material_id].flags &
                         PACKED_MATERIAL_EMISSIVE) != 0) {
                        append_mesh_emitter(instance_id, slot, material_id);
                    }
                }
            }
        }
    }

    void build_light_selection_distribution() {
        if (m_scene.non_delta_light_indices.empty()) {
            return;
        }
        std::vector<double> weights;
        weights.reserve(m_scene.non_delta_light_indices.size());
        double total = 0.0;
        for (std::uint32_t light_id : m_scene.non_delta_light_indices) {
            const double weight =
                packed_luminance(m_scene.lights[light_id].power);
            weights.push_back(weight);
            total += weight;
        }
        const double uniform =
            1.0 / static_cast<double>(weights.size());
        float cumulative = 0.0f;
        for (double weight : weights) {
            const double power_probability =
                total > 0.0 ? weight / total : uniform;
            const float probability = static_cast<float>(
                0.95 * power_probability + 0.05 * uniform);
            m_scene.light_selection_probabilities.push_back(probability);
            cumulative += probability;
            m_scene.light_cdf.push_back(cumulative);
        }
        m_scene.light_cdf.back() = 1.0f;
    }

    void compile_lights() {
        compile_explicit_lights();
        compile_auto_emitters();
        build_light_selection_distribution();
    }

    void compile_primitive(ObjectIRId id, const ObjectIRNode &node,
                           const Transform &transform, bool flip,
                           AggregateBuild &aggregate) {
        std::visit(
            [&](const auto &typed) {
                using T = std::decay_t<decltype(typed)>;
                if constexpr (std::is_same_v<T, SphereObjectIR>) {
                    PackedSphere sphere;
                    sphere.center = pack_vec3(typed.center);
                    sphere.radius = checked_float(std::abs(typed.radius),
                                                  "sphere radius");
                    if (typed.radius < 0.0) {
                        sphere.flags |= PACKED_SPHERE_FLIP_ORIENTATION;
                    }
                    const std::uint32_t sphere_id = checked_index(
                        m_scene.spheres.size(), "sphere count");
                    m_scene.spheres.push_back(sphere);
                    add_instance(
                        PackedGeometryType::Sphere, sphere_id, transform,
                        {material(typed.material, node.context)}, flip,
                        checked_index(id, "source object id"),
                        sphere_bounds(typed.center, typed.radius), aggregate);
                } else if constexpr (std::is_same_v<T,
                                                    MovingSphereObjectIR>) {
                    PackedMovingSphere sphere;
                    sphere.center0 = pack_vec3(typed.center0);
                    sphere.center1 = pack_vec3(typed.center1);
                    sphere.time0 = checked_float(typed.time0, "sphere time0");
                    sphere.time1 = checked_float(typed.time1, "sphere time1");
                    sphere.radius = checked_float(std::abs(typed.radius),
                                                  "sphere radius");
                    if (typed.radius < 0.0) {
                        sphere.flags |= PACKED_SPHERE_FLIP_ORIENTATION;
                    }
                    const std::uint32_t sphere_id = checked_index(
                        m_scene.moving_spheres.size(), "moving sphere count");
                    m_scene.moving_spheres.push_back(sphere);
                    add_instance(
                        PackedGeometryType::MovingSphere, sphere_id,
                        transform,
                        {material(typed.material, node.context)}, flip,
                        checked_index(id, "source object id"),
                        moving_sphere_bounds(typed, m_ir.time0, m_ir.time1),
                        aggregate);
                } else if constexpr (std::is_same_v<T, BoxObjectIR>) {
                    const MaterialHandle mat =
                        material(typed.material, node.context);
                    const point3 &a = typed.minimum;
                    const point3 &b = typed.maximum;
                    aggregate.generated.add_quad(
                        point3(a.x(), a.y(), b.z()),
                        vec3(b.x() - a.x(), 0, 0),
                        vec3(0, b.y() - a.y(), 0), false, mat, transform,
                        flip, checked_index(id, "source object id"));
                    aggregate.generated.add_quad(
                        point3(a.x(), a.y(), a.z()),
                        vec3(b.x() - a.x(), 0, 0),
                        vec3(0, b.y() - a.y(), 0), false, mat, transform,
                        flip, checked_index(id, "source object id"));
                    aggregate.generated.add_quad(
                        point3(a.x(), b.y(), a.z()),
                        vec3(b.x() - a.x(), 0, 0),
                        vec3(0, 0, b.z() - a.z()), true, mat, transform,
                        flip, checked_index(id, "source object id"));
                    aggregate.generated.add_quad(
                        point3(a.x(), a.y(), a.z()),
                        vec3(b.x() - a.x(), 0, 0),
                        vec3(0, 0, b.z() - a.z()), true, mat, transform,
                        flip, checked_index(id, "source object id"));
                    aggregate.generated.add_quad(
                        point3(b.x(), a.y(), a.z()),
                        vec3(0, b.y() - a.y(), 0),
                        vec3(0, 0, b.z() - a.z()), false, mat, transform,
                        flip, checked_index(id, "source object id"));
                    aggregate.generated.add_quad(
                        point3(a.x(), a.y(), a.z()),
                        vec3(0, b.y() - a.y(), 0),
                        vec3(0, 0, b.z() - a.z()), false, mat, transform,
                        flip, checked_index(id, "source object id"));
                } else if constexpr (std::is_same_v<T, AxisRectObjectIR>) {
                    const MaterialHandle mat =
                        material(typed.material, node.context);
                    if (typed.plane == AxisRectPlane::XY) {
                        aggregate.generated.add_quad(
                            point3(typed.a0, typed.b0, typed.k),
                            vec3(typed.a1 - typed.a0, 0, 0),
                            vec3(0, typed.b1 - typed.b0, 0), false, mat,
                            transform, flip,
                            checked_index(id, "source object id"));
                    } else if (typed.plane == AxisRectPlane::XZ) {
                        aggregate.generated.add_quad(
                            point3(typed.a0, typed.k, typed.b0),
                            vec3(typed.a1 - typed.a0, 0, 0),
                            vec3(0, 0, typed.b1 - typed.b0), true, mat,
                            transform, flip,
                            checked_index(id, "source object id"));
                    } else {
                        aggregate.generated.add_quad(
                            point3(typed.k, typed.a0, typed.b0),
                            vec3(0, typed.a1 - typed.a0, 0),
                            vec3(0, 0, typed.b1 - typed.b0), false, mat,
                            transform, flip,
                            checked_index(id, "source object id"));
                    }
                } else if constexpr (std::is_same_v<T, QuadObjectIR>) {
                    aggregate.generated.add_quad(
                        typed.origin, typed.u, typed.v, false,
                        material(typed.material, node.context), transform,
                        flip, checked_index(id, "source object id"));
                } else if constexpr (std::is_same_v<T, TriangleObjectIR>) {
                    aggregate.generated.add_triangle(
                        typed.positions, typed.normals, typed.uv0,
                        typed.has_normals, typed.has_uv0,
                        material(typed.material, node.context), transform,
                        flip, checked_index(id, "source object id"));
                } else if constexpr (std::is_same_v<T, ObjObjectIR>) {
                    compile_obj(typed, node, transform, flip, aggregate);
                } else if constexpr (std::is_same_v<T, ModelObjectIR>) {
                    compile_model(typed, node, transform, flip, aggregate);
                } else {
                    throw std::runtime_error(
                        "Scene compile error: expected a primitive object");
                }
            },
            node.data);
    }

    void compile_object(ObjectIRId id, const Transform &transform, bool flip,
                        AggregateBuild &aggregate, bool auto_emitters) {
        (void)auto_emitters;
        if (id >= m_ir.object_nodes.size()) {
            throw std::runtime_error(
                "Scene compile error: object reference is out of range");
        }
        if (!m_active_objects.insert(id).second) {
            throw std::runtime_error(
                "Scene compile error: object graph contains a cycle");
        }
        const ObjectIRNode &node = m_ir.object_nodes[id];
        const ObjectIRId previous_source = m_current_source;
        m_current_source = id;
        try {
            if (const auto *transformed =
                    std::get_if<TransformObjectIR>(&node.data)) {
                compile_object(transformed->child,
                               transform * transformed->transform, flip,
                               aggregate, auto_emitters);
            } else if (const auto *flipped =
                           std::get_if<FlipFaceObjectIR>(&node.data)) {
                compile_object(flipped->child, transform, !flip, aggregate,
                               auto_emitters);
            } else if (const auto *medium =
                           std::get_if<ConstantMediumObjectIR>(&node.data)) {
                if (!(medium->density > 0.0) ||
                    !std::isfinite(medium->density)) {
                    throw std::runtime_error(
                        "Scene compile error: medium density must be positive");
                }
                const std::uint32_t boundary =
                    create_boundary_aggregate(medium->boundary, transform);
                TextureHandle phase_texture;
                if (medium->texture != kInvalidTextureIR) {
                    phase_texture = build_texture(
                        medium->texture, TextureSemantic::Color, m_context);
                } else {
                    phase_texture =
                        std::make_shared<SolidColorTexture>(medium->albedo);
                }
                const MaterialHandle phase =
                    make_isotropic_material(std::move(phase_texture));
                PackedMedium packed;
                packed.boundary_aggregate = boundary;
                packed.phase_material =
                    m_resources.compile_material(phase).value;
                packed.neg_inv_density = checked_float(
                    -1.0 / medium->density, "medium inverse density");
                const std::uint32_t medium_id =
                    checked_index(m_scene.media.size(), "medium count");
                m_scene.media.push_back(packed);
                add_instance(PackedGeometryType::Medium, medium_id,
                             Transform(), {phase}, flip,
                             checked_index(id, "source object id"),
                             m_aggregate_bounds[boundary], aggregate);
            } else if (const auto *group =
                           std::get_if<GroupObjectIR>(&node.data)) {
                for (ObjectIRId child : group->children) {
                    compile_object(child, transform, flip, aggregate,
                                   auto_emitters);
                }
            } else {
                compile_primitive(id, node, transform, flip, aggregate);
            }
        } catch (...) {
            m_current_source = previous_source;
            m_active_objects.erase(id);
            throw;
        }
        m_current_source = previous_source;
        m_active_objects.erase(id);
    }

    std::uint32_t current_source_id() const {
        return checked_index(m_current_source, "source object id");
    }

    const SceneIR &m_ir;
    CompiledScene m_scene;
    SceneBuildContext m_context;
    PackedResourceCompiler m_resources;
    std::unordered_map<const MeshAsset *, std::uint32_t> m_meshes;
    std::vector<aabb> m_instance_bounds;
    std::vector<aabb> m_aggregate_bounds;
    std::unordered_set<ObjectIRId> m_active_objects;
    ObjectIRId m_current_source = kInvalidObjectIR;
};

} // namespace

CompiledScene compile_scene(const SceneIR &ir) {
    CompiledScene scene = SceneCompiler(ir).compile();
    const ValidationReport validation = validate_compiled_scene(scene);
    if (!validation.ok()) {
        throw std::runtime_error("Compiled scene validation failed: " +
                                 validation.errors.front());
    }
    return scene;
}

CompiledScene load_compiled_scene(const std::string &path) {
    return compile_scene(parse_scene_ir(load_scene_description(path)));
}

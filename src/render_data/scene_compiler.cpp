#include "scene_compiler.h"

#include "material_programs.h"
#include "model_asset.h"
#include "packed_bvh.h"
#include "resource_compiler.h"
#include "scene_loader.h"
#include "scene_loader_internal.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
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

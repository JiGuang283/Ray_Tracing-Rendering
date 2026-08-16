#include "scene_compiler_internal.h"

namespace scene_compiler_detail {

std::uint32_t SceneCompiler::pack_mesh(const MeshAsset &asset)
{
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
            if ((source.flags & MESH_TRIANGLE_REVERSE_EMITTER_NORMAL) != 0) {
                triangle.flags |= PACKED_TRIANGLE_REVERSE_EMITTER_NORMAL;
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

std::uint32_t SceneCompiler::pack_shared_mesh(
        const std::shared_ptr<const MeshAsset> &asset)
{
        auto found = m_meshes.find(asset.get());
        if (found != m_meshes.end()) {
            return found->second;
        }
        const std::uint32_t id = pack_mesh(*asset);
        m_meshes.emplace(asset.get(), id);
        return id;
    }

void SceneCompiler::flush_generated(AggregateBuild &aggregate)
{
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

void SceneCompiler::finalize_aggregate(std::uint32_t aggregate_id,
                            AggregateBuild &aggregate)
{
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

std::uint32_t SceneCompiler::create_boundary_aggregate(ObjectIRId boundary,
                                            const Transform &transform)
{
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

void SceneCompiler::compile_obj(const ObjObjectIR &object, const ObjectIRNode &node,
                     const Transform &parent, bool flip,
                     AggregateBuild &aggregate)
{
        const MaterialHandle object_material =
            material(object.material, node.context);
        const std::string path =
            resolve_asset_path(m_context.source_path, object.path);
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

void SceneCompiler::compile_model(const ModelObjectIR &object,
                       const ObjectIRNode &node, const Transform &parent,
                       bool flip, AggregateBuild &aggregate)
{
        const std::string path =
            resolve_asset_path(m_context.source_path, object.path);
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

void SceneCompiler::compile_model_node(const ModelAsset &asset, std::size_t node_index,
                            const Transform &parent,
                            const std::vector<MaterialHandle> &materials,
                            bool flip, AggregateBuild &aggregate)
{
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

} // namespace scene_compiler_detail

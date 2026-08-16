#include "scene_compiler_internal.h"

namespace scene_compiler_detail {

SceneCompiler::SceneCompiler(const SceneIR &ir)
        : m_ir(ir), m_resources(m_scene)
{
        m_context.source_path = ir.source_path;
        m_context.scene_ir = &ir;
        m_context.strict_assets = ir.strict_assets;
        m_context.resources.set_strict_assets(ir.strict_assets);
        m_scene.camera =
            compile_packed_camera(ir.camera, ir.time0, ir.time1);
        m_scene.background = pack_vec3(ir.preset.background);
        m_scene.scene_time0 = checked_float(ir.time0, "scene time0");
        m_scene.scene_time1 = checked_float(ir.time1, "scene time1");
    }

CompiledScene SceneCompiler::compile()
{
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

MaterialHandle SceneCompiler::material(const std::string &name,
                            const std::string &context)
{
        return lookup_material(m_context, name, context);
    }

std::uint32_t SceneCompiler::add_transform(const Transform &transform)
{
        const std::uint32_t id =
            checked_index(m_scene.transforms.size(), "transform count");
        m_scene.transforms.push_back(pack_transform(transform));
        return id;
    }

Range32 SceneCompiler::add_material_bindings(
        const std::vector<MaterialHandle> &materials)
{
        const std::size_t offset = m_scene.material_bindings.size();
        for (const MaterialHandle &material_handle : materials) {
            m_scene.material_bindings.push_back(
                m_resources.compile_material(material_handle).value);
            m_scene.emitter_bindings.push_back(kInvalidPackedIndex);
        }
        return checked_range(offset, materials.size(), "material bindings");
    }

std::uint32_t SceneCompiler::add_instance(PackedGeometryType type,
                               std::uint32_t geometry_index,
                               const Transform &transform,
                               const std::vector<MaterialHandle> &materials,
                               bool flip, std::uint32_t source_object_id,
                               const aabb &object_bounds,
                               AggregateBuild &aggregate)
{
        const aabb world_bounds = transform.bounds_to_world(object_bounds);
        const PackedBVHNode packed_bounds = pack_packed_bounds(world_bounds);
        PackedInstance instance;
        instance.geometry_type = type;
        instance.geometry_index = geometry_index;
        instance.transform_id = add_transform(transform);
        instance.flags = flip ? PACKED_INSTANCE_FLIP_FACE
                              : PACKED_INSTANCE_NONE;
        if (transform_is_identity(transform)) {
            instance.flags |= PACKED_INSTANCE_HOST_IDENTITY_TRANSFORM;
        }
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

} // namespace scene_compiler_detail

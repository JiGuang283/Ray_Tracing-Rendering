#include "scene_loader_internal.h"

#include "asset_path.h"
#include "accel.h"
#include "aarect.h"
#include "box.h"
#include "constant_medium.h"
#include "mesh_instance.h"
#include "mesh_light.h"
#include "model_asset.h"
#include "moving_sphere.h"
#include "quad_light.h"
#include "sphere.h"
#include "sphere_light.h"
#include "transformed_hittable.h"
#include "transformed_light.h"
#include "triangle.h"
#include "triangle_light.h"

#include <stdexcept>
#include <type_traits>
#include <utility>

namespace scene_loader_internal {
namespace {

const ObjectIRNode &object_node(ObjectIRId id,
                                const SceneBuildContext &context) {
    if (!context.scene_ir || id >= context.scene_ir->object_nodes.size()) {
        throw std::runtime_error(
            "Scene build error: invalid ObjectIR reference.");
    }
    return context.scene_ir->object_nodes[id];
}

void append_emitter(BuiltObject &built, shared_ptr<Light> emitter) {
    if (emitter) {
        built.emitters.push_back(std::move(emitter));
    }
}

BuiltObject build_object_impl(ObjectIRId id, SceneBuildContext &context,
                              bool auto_emitters, bool flip_emitters);

BuiltObject build_obj(const ObjObjectIR &obj, const std::string &node_context,
                      SceneBuildContext &context, bool auto_emitters,
                      bool flip_emitters) {
    const MaterialHandle material =
        lookup_material(context, obj.material, node_context);
    const std::string path =
        resolve_asset_path(context.source_path, obj.path);
    std::string error;
    const std::shared_ptr<const MeshAsset> asset =
        context.resources.load_obj(path, obj.build_bvh,
                                   obj.use_vertex_normals, error);
    if (!asset) {
        throw std::runtime_error("Scene file error: failed to load OBJ '" +
                                 path + "': " + error);
    }

    Transform object_to_world =
        Transform::rotate_y(obj.rotation_y) *
        Transform::translate(obj.local_translation) *
        Transform::scale(obj.scale);
    vec3 position = obj.position;
    if (obj.auto_lift_to_ground) {
        const aabb transformed_bounds =
            object_to_world.bounds_to_world(asset->bounds());
        position[1] -= transformed_bounds.min().y();
        object_to_world = Transform::translate(position) * object_to_world;
    } else if (obj.has_position) {
        object_to_world = Transform::translate(position) * object_to_world;
    }

    auto instance = make_shared<MeshInstance>(
        asset, std::vector<MaterialHandle>{material}, object_to_world);
    BuiltObject built;
    built.object = instance;
    if (auto_emitters && material->is_emissive()) {
        append_emitter(
            built, make_shared<MeshLight>(instance, 0, flip_emitters));
    }
    return built;
}

void instantiate_model_node(const ModelAsset &asset, std::size_t node_index,
                            const Transform &parent_to_world,
                            const std::vector<MaterialHandle> &materials,
                            hittable_list &instances, BuiltObject &built,
                            bool auto_emitters, bool flip_emitters) {
    if (node_index >= asset.nodes().size()) {
        throw std::runtime_error(
            "Scene build error: model contains an invalid node reference.");
    }
    const ModelNode &node = asset.nodes()[node_index];
    const Transform node_to_world = parent_to_world * node.local_transform;
    if (node.mesh_index >= 0) {
        const ModelMesh &mesh =
            asset.meshes()[static_cast<std::size_t>(node.mesh_index)];
        auto instance = make_shared<MeshInstance>(
            mesh.geometry, materials, node_to_world);
        instances.add(instance);

        if (auto_emitters) {
            std::vector<bool> used_slots(materials.size(), false);
            for (std::size_t triangle_index = 0;
                 triangle_index < mesh.geometry->triangle_count();
                 ++triangle_index) {
                const std::size_t slot =
                    mesh.geometry->triangle(
                        static_cast<std::uint32_t>(triangle_index))
                        .material_slot;
                if (used_slots[slot]) {
                    continue;
                }
                used_slots[slot] = true;
                if (!materials[slot]->is_emissive()) {
                    continue;
                }
                append_emitter(built,
                               make_shared<MeshLight>(
                                   instance, static_cast<std::uint32_t>(slot),
                                   flip_emitters));
            }
        }
    }
    for (std::size_t child : node.children) {
        instantiate_model_node(asset, child, node_to_world, materials,
                               instances, built, auto_emitters,
                               flip_emitters);
    }
}

BuiltObject build_model(const ModelObjectIR &model,
                        const std::string &node_context,
                        SceneBuildContext &context, bool auto_emitters,
                        bool flip_emitters) {
    const std::string path =
        resolve_asset_path(context.source_path, model.path);
    ModelImportOptions options;
    std::string error;
    const std::shared_ptr<const ModelAsset> asset =
        context.resources.load_model(path, options, error);
    if (!asset) {
        throw std::runtime_error("Scene file error: failed to load model '" +
                                 path + "': " + error);
    }

    std::vector<MaterialHandle> materials = asset->materials();
    for (const auto &override_entry : model.material_overrides) {
        bool matched = false;
        for (std::size_t slot = 0; slot < asset->material_names().size();
             ++slot) {
            if (asset->material_names()[slot] == override_entry.first) {
                materials[slot] = lookup_material(
                    context, override_entry.second,
                    node_context + ".material_overrides." +
                        override_entry.first);
                matched = true;
            }
        }
        if (!matched) {
            throw std::runtime_error(
                "Scene file error: model material override references unknown "
                "material '" +
                override_entry.first + "' in " + node_context + ".");
        }
    }

    const int scene_index = asset->resolve_scene_index(model.scene_index);
    const ModelScene &scene =
        asset->scenes()[static_cast<std::size_t>(scene_index)];
    hittable_list instances;
    BuiltObject built;
    for (std::size_t root : scene.roots) {
        instantiate_model_node(*asset, root, model.transform, materials,
                               instances, built, auto_emitters,
                               flip_emitters);
    }
    if (instances.objects.empty()) {
        throw std::runtime_error("Scene file error: selected model scene '" +
                                 scene.name + "' contains no mesh instances.");
    }
    built.object = instances.objects.size() == 1
                       ? instances.objects.front()
                       : make_accel(instances);
    return built;
}

BuiltObject build_primitive(const ObjectIRNode &node,
                            SceneBuildContext &context,
                            bool auto_emitters, bool flip_emitters) {
    return std::visit(
        [&](const auto &typed) -> BuiltObject {
            using T = std::decay_t<decltype(typed)>;
            BuiltObject built;

            if constexpr (std::is_same_v<T, SphereObjectIR>) {
                const MaterialHandle material =
                    lookup_material(context, typed.material, node.context);
                built.object = make_shared<sphere>(typed.center, typed.radius,
                                                   material);
                if (auto_emitters && !flip_emitters &&
                    material->is_emissive()) {
                    append_emitter(
                        built,
                        make_shared<SphereLight>(typed.center, typed.radius,
                                                 material));
                }
            } else if constexpr (std::is_same_v<T, MovingSphereObjectIR>) {
                built.object = make_shared<moving_sphere>(
                    typed.center0, typed.center1, typed.time0, typed.time1,
                    typed.radius,
                    lookup_material(context, typed.material, node.context));
            } else if constexpr (std::is_same_v<T, BoxObjectIR>) {
                built.object = make_shared<box>(
                    typed.minimum, typed.maximum,
                    lookup_material(context, typed.material, node.context));
            } else if constexpr (std::is_same_v<T, AxisRectObjectIR>) {
                const MaterialHandle material =
                    lookup_material(context, typed.material, node.context);
                if (typed.plane == AxisRectPlane::XY) {
                    built.object = make_shared<xy_rect>(
                        typed.a0, typed.a1, typed.b0, typed.b1, typed.k,
                        material);
                    const point3 origin(typed.a0, typed.b0, typed.k);
                    const vec3 u(typed.a1 - typed.a0, 0, 0);
                    const vec3 v(0, typed.b1 - typed.b0, 0);
                    if (auto_emitters && material->is_emissive()) {
                        append_emitter(
                            built,
                            make_shared<QuadLight>(
                                origin, flip_emitters ? v : u,
                                flip_emitters ? u : v, material, true,
                                flip_emitters));
                    }
                } else if (typed.plane == AxisRectPlane::XZ) {
                    built.object = make_shared<xz_rect>(
                        typed.a0, typed.a1, typed.b0, typed.b1, typed.k,
                        material);
                    const point3 origin(typed.a0, typed.k, typed.b0);
                    const vec3 u(typed.a1 - typed.a0, 0, 0);
                    const vec3 v(0, 0, typed.b1 - typed.b0);
                    if (auto_emitters && material->is_emissive()) {
                        append_emitter(
                            built,
                            make_shared<QuadLight>(
                                origin, flip_emitters ? v : u,
                                flip_emitters ? u : v, material, true,
                                flip_emitters));
                    }
                } else {
                    built.object = make_shared<yz_rect>(
                        typed.a0, typed.a1, typed.b0, typed.b1, typed.k,
                        material);
                    const point3 origin(typed.k, typed.a0, typed.b0);
                    const vec3 u(0, typed.a1 - typed.a0, 0);
                    const vec3 v(0, 0, typed.b1 - typed.b0);
                    if (auto_emitters && material->is_emissive()) {
                        append_emitter(
                            built,
                            make_shared<QuadLight>(
                                origin, flip_emitters ? v : u,
                                flip_emitters ? u : v, material, true,
                                flip_emitters));
                    }
                }
            } else if constexpr (std::is_same_v<T, QuadObjectIR>) {
                const MaterialHandle material =
                    lookup_material(context, typed.material, node.context);
                auto quad = make_shared<hittable_list>();
                quad->add(make_shared<triangle>(
                    typed.origin, typed.origin + typed.u,
                    typed.origin + typed.u + typed.v, material, vec2(0, 0),
                    vec2(1, 0), vec2(1, 1), true));
                quad->add(make_shared<triangle>(
                    typed.origin, typed.origin + typed.u + typed.v,
                    typed.origin + typed.v, material, vec2(0, 0), vec2(1, 1),
                    vec2(0, 1), true));
                built.object = quad;
                if (auto_emitters && material->is_emissive()) {
                    append_emitter(
                        built,
                        make_shared<QuadLight>(
                            typed.origin, flip_emitters ? typed.v : typed.u,
                            flip_emitters ? typed.u : typed.v, material, true,
                            flip_emitters));
                }
            } else if constexpr (std::is_same_v<T, TriangleObjectIR>) {
                const MaterialHandle material =
                    lookup_material(context, typed.material, node.context);
                if (typed.has_normals) {
                    built.object = make_shared<triangle>(
                        typed.positions[0], typed.positions[1],
                        typed.positions[2], typed.normals[0], typed.normals[1],
                        typed.normals[2], material, typed.uv0[0], typed.uv0[1],
                        typed.uv0[2], typed.has_uv0);
                } else {
                    built.object = make_shared<triangle>(
                        typed.positions[0], typed.positions[1],
                        typed.positions[2], material, typed.uv0[0],
                        typed.uv0[1], typed.uv0[2], typed.has_uv0);
                }
                if (auto_emitters && material->is_emissive()) {
                    const std::size_t first = 0;
                    const std::size_t second = flip_emitters ? 2 : 1;
                    const std::size_t third = flip_emitters ? 1 : 2;
                    append_emitter(
                        built,
                        make_shared<TriangleLight>(
                            typed.positions[first], typed.positions[second],
                            typed.positions[third], material, typed.uv0[first],
                            typed.uv0[second], typed.uv0[third], typed.has_uv0));
                }
            } else if constexpr (std::is_same_v<T, ObjObjectIR>) {
                return build_obj(typed, node.context, context, auto_emitters,
                                 flip_emitters);
            } else if constexpr (std::is_same_v<T, ModelObjectIR>) {
                return build_model(typed, node.context, context,
                                   auto_emitters, flip_emitters);
            } else {
                throw std::runtime_error(
                    "Scene build error: expected a primitive ObjectIR node in " +
                    node.context + ".");
            }
            return built;
        },
        node.data);
}

BuiltObject build_object_impl(ObjectIRId id, SceneBuildContext &context,
                              bool auto_emitters, bool flip_emitters) {
    const ObjectIRNode &node = object_node(id, context);
    if (const auto *transformed =
            std::get_if<TransformObjectIR>(&node.data)) {
        BuiltObject child = build_object_impl(
            transformed->child, context, auto_emitters, flip_emitters);
        child.object = make_shared<TransformedHittable>(
            child.object, transformed->transform);
        for (shared_ptr<Light> &emitter : child.emitters) {
            emitter = make_shared<TransformedLight>(
                emitter, transformed->transform);
        }
        return child;
    }
    if (const auto *flipped = std::get_if<FlipFaceObjectIR>(&node.data)) {
        BuiltObject child = build_object_impl(
            flipped->child, context, auto_emitters, !flip_emitters);
        child.object = make_shared<flip_face>(child.object);
        return child;
    }
    if (const auto *medium =
            std::get_if<ConstantMediumObjectIR>(&node.data)) {
        const shared_ptr<hittable> boundary =
            build_object_impl(medium->boundary, context, false, false).object;
        BuiltObject built;
        if (medium->texture != kInvalidTextureIR) {
            built.object = make_shared<constant_medium>(
                boundary, medium->density,
                build_texture(medium->texture, TextureSemantic::Color,
                              context));
        } else {
            built.object = make_shared<constant_medium>(
                boundary, medium->density, medium->albedo);
        }
        return built;
    }
    if (const auto *group = std::get_if<GroupObjectIR>(&node.data)) {
        hittable_list objects;
        BuiltObject built;
        for (ObjectIRId child_id : group->children) {
            BuiltObject child = build_object_impl(
                child_id, context, auto_emitters, flip_emitters);
            objects.add(child.object);
            built.emitters.insert(built.emitters.end(),
                                  child.emitters.begin(),
                                  child.emitters.end());
        }
        built.object = group->accelerate
                           ? make_accel(objects, group->time0, group->time1)
                           : make_shared<hittable_list>(objects);
        return built;
    }
    return build_primitive(node, context, auto_emitters, flip_emitters);
}

} // namespace

shared_ptr<hittable> build_object(ObjectIRId id,
                                  SceneBuildContext &context) {
    return build_object_impl(id, context, false, false).object;
}

BuiltObject build_object_with_emitters(ObjectIRId id,
                                       SceneBuildContext &context,
                                       bool auto_emitters) {
    return build_object_impl(id, context, auto_emitters, false);
}

void add_object(ObjectIRId id, SceneBuildContext &context,
                hittable_list &world,
                std::vector<shared_ptr<Light>> &emitters,
                bool auto_emitters) {
    BuiltObject built =
        build_object_with_emitters(id, context, auto_emitters);
    world.add(built.object);
    emitters.insert(emitters.end(), built.emitters.begin(),
                    built.emitters.end());
}

} // namespace scene_loader_internal

#include "scene_compiler_internal.h"

namespace scene_compiler_detail {

void SceneCompiler::compile_primitive(ObjectIRId id, const ObjectIRNode &node,
                           const Transform &transform, bool flip,
                           AggregateBuild &aggregate)
{
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
                            checked_index(id, "source object id"), true);
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

void SceneCompiler::compile_object(ObjectIRId id, const Transform &transform, bool flip,
                        AggregateBuild &aggregate, bool auto_emitters)
{
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

std::uint32_t SceneCompiler::current_source_id() const
{
        return checked_index(m_current_source, "source object id");
    }

} // namespace scene_compiler_detail

#include "scene_loader_internal.h"

#include "accel.h"
#include "aarect.h"
#include "box.h"
#include "constant_medium.h"
#include "mesh_instance.h"
#include "moving_sphere.h"
#include "obj_importer.h"
#include "quad_light.h"
#include "sphere.h"
#include "sphere_light.h"
#include "transformed_light.h"
#include "triangle.h"
#include "triangle_light.h"
#include "mesh_light.h"

#include <stdexcept>

namespace scene_loader_internal {

namespace {

shared_ptr<MeshInstance>
load_obj_instance(const json &object, SceneBuildContext &context,
                  const MaterialHandle &material) {
    const std::string path = resolve_asset_path(
        context, read_string(object, "path", "obj"));
    ObjImportOptions options;
    options.build_bvh = read_bool_or(object, "build_bvh", true);
    options.use_vertex_normals =
        read_bool_or(object, "use_vertex_normals", true);
    std::string error;
    std::shared_ptr<const MeshAsset> asset =
        load_obj_mesh_asset(path, options, error);
    if (!asset) {
        throw std::runtime_error("Scene file error: failed to load OBJ '" +
                                 path + "': " + error);
    }

    const Transform local_transform =
        Transform::translate(read_vec3_or(object, "translate", vec3(0, 0, 0),
                                          "obj")) *
        Transform::scale(
            read_vec3_or(object, "scale", vec3(1, 1, 1), "obj"));
    return make_shared<MeshInstance>(asset,
                                     std::vector<MaterialHandle>{material},
                                     local_transform);
}

std::vector<TriangleSurface>
flip_triangles(const std::vector<TriangleSurface> &triangles) {
    std::vector<TriangleSurface> result;
    result.reserve(triangles.size());
    for (const auto &triangle : triangles) {
        result.push_back({triangle.v0, triangle.v2, triangle.v1});
    }
    return result;
}

void add_area_emitter_if_needed(BuiltObject &built, const json &object,
                                const MaterialHandle &mat,
                                bool flip_emitters) {
    if (!mat || !mat->is_emissive()) {
        return;
    }

    std::string type = read_string(object, "type", "object");
    color emission = mat->emission_estimate();
    if (emission.length_squared() <= 0.0) {
        return;
    }

    if (type == "xy_rect") {
        double x0 = read_double_or(object, "x0", 0.0);
        double x1 = read_double_or(object, "x1", 1.0);
        double y0 = read_double_or(object, "y0", 0.0);
        double y1 = read_double_or(object, "y1", 1.0);
        double k = read_double_or(object, "k", 0.0);
        vec3 u(x1 - x0, 0, 0);
        vec3 v(0, y1 - y0, 0);
        built.emitters.push_back(flip_emitters
                                     ? make_shared<QuadLight>(
                                           point3(x0, y0, k), v, u, emission)
                                     : make_shared<QuadLight>(
                                           point3(x0, y0, k), u, v, emission));
        return;
    }
    if (type == "xz_rect") {
        double x0 = read_double_or(object, "x0", 0.0);
        double x1 = read_double_or(object, "x1", 1.0);
        double z0 = read_double_or(object, "z0", 0.0);
        double z1 = read_double_or(object, "z1", 1.0);
        double k = read_double_or(object, "k", 0.0);
        vec3 u(x1 - x0, 0, 0);
        vec3 v(0, 0, z1 - z0);
        built.emitters.push_back(flip_emitters
                                     ? make_shared<QuadLight>(
                                           point3(x0, k, z0), v, u, emission)
                                     : make_shared<QuadLight>(
                                           point3(x0, k, z0), u, v, emission));
        return;
    }
    if (type == "yz_rect") {
        double y0 = read_double_or(object, "y0", 0.0);
        double y1 = read_double_or(object, "y1", 1.0);
        double z0 = read_double_or(object, "z0", 0.0);
        double z1 = read_double_or(object, "z1", 1.0);
        double k = read_double_or(object, "k", 0.0);
        vec3 u(0, y1 - y0, 0);
        vec3 v(0, 0, z1 - z0);
        built.emitters.push_back(flip_emitters
                                     ? make_shared<QuadLight>(
                                           point3(k, y0, z0), v, u, emission)
                                     : make_shared<QuadLight>(
                                           point3(k, y0, z0), u, v, emission));
        return;
    }
    if (type == "quad") {
        point3 q = read_vec3(object, "Q", "quad");
        vec3 u = read_vec3(object, "u", "quad");
        vec3 v = read_vec3(object, "v", "quad");
        built.emitters.push_back(flip_emitters
                                     ? make_shared<QuadLight>(q, v, u,
                                                              emission)
                                     : make_shared<QuadLight>(q, u, v,
                                                              emission));
        return;
    }
    if (type == "triangle") {
        point3 v0;
        point3 v1;
        point3 v2;
        if (object.contains("vertices")) {
            const auto &vertices = object["vertices"];
            if (!vertices.is_array() || vertices.size() != 3) {
                throw std::runtime_error(
                    "Scene file error: triangle.vertices must contain 3 points.");
            }
            v0 = read_vec3_value(vertices[0], "triangle.vertices[0]");
            v1 = read_vec3_value(vertices[1], "triangle.vertices[1]");
            v2 = read_vec3_value(vertices[2], "triangle.vertices[2]");
        } else {
            v0 = read_vec3(object, "v0", "triangle");
            v1 = read_vec3(object, "v1", "triangle");
            v2 = read_vec3(object, "v2", "triangle");
        }
        built.emitters.push_back(flip_emitters
                                     ? make_shared<TriangleLight>(
                                           v0, v2, v1, emission)
                                     : make_shared<TriangleLight>(
                                           v0, v1, v2, emission));
        return;
    }
    if (type == "sphere" && !flip_emitters) {
        built.emitters.push_back(make_shared<SphereLight>(
            read_vec3(object, "center", "sphere"),
            read_double_or(object, "radius", 1.0), emission));
    }
}

std::vector<shared_ptr<Light>>
translate_emitters(const std::vector<shared_ptr<Light>> &emitters,
                   const vec3 &offset) {
    std::vector<shared_ptr<Light>> result;
    result.reserve(emitters.size());
    for (const auto &emitter : emitters) {
        result.push_back(make_shared<TranslateLight>(emitter, offset));
    }
    return result;
}

std::vector<shared_ptr<Light>>
rotate_y_emitters(const std::vector<shared_ptr<Light>> &emitters,
                  double angle) {
    std::vector<shared_ptr<Light>> result;
    result.reserve(emitters.size());
    for (const auto &emitter : emitters) {
        result.push_back(make_shared<RotateYLight>(emitter, angle));
    }
    return result;
}

BuiltObject build_obj_with_emitters(const json &object,
                                    SceneBuildContext &context,
                                    bool auto_emitters,
                                    bool flip_emitters) {
    auto mat = lookup_material(context, object, "obj");
    if (object.contains("implementation")) {
        throw std::runtime_error(
            "Scene file error: obj.implementation was removed; MeshAsset "
            "is now the only OBJ path.");
    }

    auto mesh = load_obj_instance(object, context, mat);

    BuiltObject built;
    built.object = mesh;
    if (auto_emitters && mat && mat->is_emissive()) {
        color emission = mat->emission_estimate();
        if (emission.length_squared() > 0.0) {
            auto triangles = mesh->light_triangles();
            built.emitters.push_back(make_shared<MeshLight>(
                flip_emitters ? flip_triangles(triangles) : triangles,
                emission));
        }
    }

    if (object.contains("rotation_y")) {
        double angle = read_double_or(object, "rotation_y", 0.0);
        built.object = make_shared<rotate_y>(built.object, angle);
        built.emitters = rotate_y_emitters(built.emitters, angle);
    }

    if (read_bool_or(object, "auto_lift_to_ground", false)) {
        aabb bbox;
        if (!built.object->bounding_box(0.0, 1.0, bbox)) {
            throw std::runtime_error(
                "Scene file error: cannot auto-lift OBJ without bbox.");
        }
        vec3 position = read_vec3_or(object, "position", vec3(0, 0, 0),
                                     "obj");
        position[1] += -bbox.min().y();
        built.object = make_shared<translate>(built.object, position);
        built.emitters = translate_emitters(built.emitters, position);
    } else if (object.contains("position")) {
        vec3 position = read_vec3(object, "position", "obj");
        built.object = make_shared<translate>(built.object, position);
        built.emitters = translate_emitters(built.emitters, position);
    }

    return built;
}

} // namespace

shared_ptr<hittable> build_object(const json &object,
                                  SceneBuildContext &context) {
    std::string type = read_string(object, "type", "object");

    if (type == "sphere") {
        auto mat = lookup_material(context, object, "sphere");
        return make_shared<sphere>(
            read_vec3(object, "center", "sphere"),
            read_double_or(object, "radius", 1.0), mat);
    }
    if (type == "moving_sphere") {
        auto mat = lookup_material(context, object, "moving_sphere");
        return make_shared<moving_sphere>(
            read_vec3(object, "center0", "moving_sphere"),
            read_vec3(object, "center1", "moving_sphere"),
            read_double_or(object, "time0", 0.0),
            read_double_or(object, "time1", 1.0),
            read_double_or(object, "radius", 1.0), mat);
    }
    if (type == "box") {
        auto mat = lookup_material(context, object, "box");
        return make_shared<box>(read_vec3(object, "min", "box"),
                                read_vec3(object, "max", "box"), mat);
    }
    if (type == "xy_rect") {
        auto mat = lookup_material(context, object, "xy_rect");
        return make_shared<xy_rect>(
            read_double_or(object, "x0", 0.0), read_double_or(object, "x1", 1.0),
            read_double_or(object, "y0", 0.0), read_double_or(object, "y1", 1.0),
            read_double_or(object, "k", 0.0), mat);
    }
    if (type == "xz_rect") {
        auto mat = lookup_material(context, object, "xz_rect");
        return make_shared<xz_rect>(
            read_double_or(object, "x0", 0.0), read_double_or(object, "x1", 1.0),
            read_double_or(object, "z0", 0.0), read_double_or(object, "z1", 1.0),
            read_double_or(object, "k", 0.0), mat);
    }
    if (type == "yz_rect") {
        auto mat = lookup_material(context, object, "yz_rect");
        return make_shared<yz_rect>(
            read_double_or(object, "y0", 0.0), read_double_or(object, "y1", 1.0),
            read_double_or(object, "z0", 0.0), read_double_or(object, "z1", 1.0),
            read_double_or(object, "k", 0.0), mat);
    }
    if (type == "quad") {
        auto mat = lookup_material(context, object, "quad");
        point3 q = read_vec3(object, "Q", "quad");
        vec3 u = read_vec3(object, "u", "quad");
        vec3 v = read_vec3(object, "v", "quad");
        auto quad = make_shared<hittable_list>();
        quad->add(make_shared<triangle>(q, q + u, q + u + v, mat));
        quad->add(make_shared<triangle>(q, q + u + v, q + v, mat));
        return quad;
    }
    if (type == "triangle") {
        auto mat = lookup_material(context, object, "triangle");
        point3 v0;
        point3 v1;
        point3 v2;
        if (object.contains("vertices")) {
            const auto &vertices = object["vertices"];
            if (!vertices.is_array() || vertices.size() != 3) {
                throw std::runtime_error(
                    "Scene file error: triangle.vertices must contain 3 points.");
            }
            v0 = read_vec3_value(vertices[0], "triangle.vertices[0]");
            v1 = read_vec3_value(vertices[1], "triangle.vertices[1]");
            v2 = read_vec3_value(vertices[2], "triangle.vertices[2]");
        } else {
            v0 = read_vec3(object, "v0", "triangle");
            v1 = read_vec3(object, "v1", "triangle");
            v2 = read_vec3(object, "v2", "triangle");
        }

        bool has_uv0 = false, has_uv1 = false, has_uv2 = false;
        vec2 uv0 = read_optional_uv(object, "uv0", has_uv0, "triangle");
        vec2 uv1 = read_optional_uv(object, "uv1", has_uv1, "triangle");
        vec2 uv2 = read_optional_uv(object, "uv2", has_uv2, "triangle");
        bool has_uvs = has_uv0 && has_uv1 && has_uv2;

        if (object.contains("n0") && object.contains("n1") &&
            object.contains("n2")) {
            return make_shared<triangle>(
                v0, v1, v2, unit_vector(read_vec3(object, "n0", "triangle")),
                unit_vector(read_vec3(object, "n1", "triangle")),
                unit_vector(read_vec3(object, "n2", "triangle")), mat, uv0,
                uv1, uv2, has_uvs);
        }
        return make_shared<triangle>(v0, v1, v2, mat, uv0, uv1, uv2,
                                     has_uvs);
    }
    if (type == "obj") {
        auto mat = lookup_material(context, object, "obj");
        if (object.contains("implementation")) {
            throw std::runtime_error(
                "Scene file error: obj.implementation was removed; MeshAsset "
                "is now the only OBJ path.");
        }
        shared_ptr<hittable> mesh_object =
            load_obj_instance(object, context, mat);
        if (object.contains("rotation_y")) {
            mesh_object = make_shared<rotate_y>(
                mesh_object, read_double_or(object, "rotation_y", 0.0));
        }
        if (read_bool_or(object, "auto_lift_to_ground", false)) {
            aabb bbox;
            if (!mesh_object->bounding_box(0.0, 1.0, bbox)) {
                throw std::runtime_error(
                    "Scene file error: cannot auto-lift OBJ without bbox.");
            }
            vec3 position =
                read_vec3_or(object, "position", vec3(0, 0, 0), "obj");
            position[1] += -bbox.min().y();
            mesh_object = make_shared<translate>(mesh_object, position);
        } else if (object.contains("position")) {
            mesh_object =
                make_shared<translate>(mesh_object,
                                       read_vec3(object, "position", "obj"));
        }
        return mesh_object;
    }
    if (type == "translate") {
        const auto &child = require_key(object, "object", "translate");
        vec3 offset = read_vec3_or(object, "offset",
                                   read_vec3_or(object, "translate",
                                                vec3(0, 0, 0), "translate"),
                                   "translate");
        return make_shared<translate>(build_object(child, context),
                                      offset);
    }
    if (type == "rotate_y") {
        return make_shared<rotate_y>(
            build_object(require_key(object, "object", "rotate_y"), context),
            read_double_or(object, "angle", 0.0));
    }
    if (type == "flip_face") {
        return make_shared<flip_face>(
            build_object(require_key(object, "object", "flip_face"),
                         context));
    }
    if (type == "constant_medium") {
        auto boundary =
            build_object(object.contains("boundary")
                             ? object["boundary"]
                             : require_key(object, "object",
                                           "constant_medium"),
                         context);
        double density = read_double_or(object, "density", 1.0);
        if (object.contains("_texture_ir_id")) {
            return make_shared<constant_medium>(
                boundary, density,
                build_texture(object["_texture_ir_id"].get<TextureIRId>(),
                              TextureSemantic::Color, context));
        }
        return make_shared<constant_medium>(
            boundary, density,
            read_vec3_or(object, "color", color(1, 1, 1),
                         "constant_medium"));
    }
    if (type == "list" || type == "accel") {
        const auto &children = require_key(object, "objects", type);
        if (!children.is_array()) {
            throw std::runtime_error("Scene file error: '" + type +
                                     ".objects' must be an array.");
        }
        hittable_list list;
        for (const auto &child : children) {
            list.add(build_object(child, context));
        }
        if (type == "accel" || read_bool_or(object, "accelerate", false)) {
            return make_accel(list, read_double_or(object, "time0", 0.0),
                              read_double_or(object, "time1", 1.0));
        }
        return make_shared<hittable_list>(list);
    }

    throw std::runtime_error("Scene file error: unknown object type '" + type +
                             "'.");
}

namespace {

BuiltObject build_object_with_emitters_impl(const json &object,
                                            SceneBuildContext &context,
                                            bool auto_emitters,
                                            bool flip_emitters) {
    std::string type = read_string(object, "type", "object");

    if (type == "translate") {
        const auto &child = require_key(object, "object", "translate");
        vec3 offset = read_vec3_or(object, "offset",
                                   read_vec3_or(object, "translate",
                                                vec3(0, 0, 0), "translate"),
                                   "translate");
        BuiltObject child_built = build_object_with_emitters_impl(
            child, context, auto_emitters, flip_emitters);
        BuiltObject built;
        built.object = make_shared<translate>(child_built.object, offset);
        built.emitters = translate_emitters(child_built.emitters, offset);
        return built;
    }

    if (type == "rotate_y") {
        double angle = read_double_or(object, "angle", 0.0);
        BuiltObject child_built = build_object_with_emitters_impl(
            require_key(object, "object", "rotate_y"), context, auto_emitters,
            flip_emitters);
        BuiltObject built;
        built.object = make_shared<rotate_y>(child_built.object, angle);
        built.emitters = rotate_y_emitters(child_built.emitters, angle);
        return built;
    }

    if (type == "flip_face") {
        BuiltObject child_built = build_object_with_emitters_impl(
            require_key(object, "object", "flip_face"), context,
            auto_emitters, !flip_emitters);
        BuiltObject built;
        built.object = make_shared<flip_face>(child_built.object);
        built.emitters = child_built.emitters;
        return built;
    }

    if (type == "list" || type == "accel") {
        const auto &children = require_key(object, "objects", type);
        if (!children.is_array()) {
            throw std::runtime_error("Scene file error: '" + type +
                                     ".objects' must be an array.");
        }
        hittable_list list;
        BuiltObject built;
        for (const auto &child : children) {
            BuiltObject child_built = build_object_with_emitters_impl(
                child, context, auto_emitters, flip_emitters);
            list.add(child_built.object);
            built.emitters.insert(built.emitters.end(),
                                  child_built.emitters.begin(),
                                  child_built.emitters.end());
        }
        if (type == "accel" || read_bool_or(object, "accelerate", false)) {
            built.object = make_accel(list, read_double_or(object, "time0", 0.0),
                                      read_double_or(object, "time1", 1.0));
        } else {
            built.object = make_shared<hittable_list>(list);
        }
        return built;
    }

    if (type == "obj") {
        return build_obj_with_emitters(object, context, auto_emitters,
                                       flip_emitters);
    }

    BuiltObject built;
    built.object = build_object(object, context);

    if (auto_emitters &&
        (type == "xy_rect" || type == "xz_rect" || type == "yz_rect" ||
         type == "quad" || type == "triangle" || type == "sphere")) {
        add_area_emitter_if_needed(
            built, object, lookup_material(context, object, type),
            flip_emitters);
    }
    return built;
}

} // namespace

BuiltObject build_object_with_emitters(const json &object,
                                       SceneBuildContext &context,
                                       bool auto_emitters) {
    return build_object_with_emitters_impl(object, context, auto_emitters,
                                           false);
}

void add_object(const json &object, SceneBuildContext &context,
                hittable_list &world,
                std::vector<shared_ptr<Light>> &emitters,
                bool auto_emitters) {
    BuiltObject built =
        build_object_with_emitters(object, context, auto_emitters);
    world.add(built.object);
    emitters.insert(emitters.end(), built.emitters.begin(),
                    built.emitters.end());
}


} // namespace scene_loader_internal

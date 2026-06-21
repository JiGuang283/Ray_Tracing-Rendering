#include "scene_loader_internal.h"

#include "accel.h"
#include "aarect.h"
#include "box.h"
#include "constant_medium.h"
#include "mesh.h"
#include "moving_sphere.h"
#include "sphere.h"
#include "triangle.h"

#include <stdexcept>

namespace scene_loader_internal {

shared_ptr<hittable> build_object(const json &object,
                                  const MaterialMap &materials,
                                  const TextureMap &textures) {
    std::string type = read_string(object, "type", "object");

    if (type == "sphere") {
        auto mat = lookup_material(materials, object, "sphere");
        return make_shared<sphere>(
            read_vec3(object, "center", "sphere"),
            read_double_or(object, "radius", 1.0), mat);
    }
    if (type == "moving_sphere") {
        auto mat = lookup_material(materials, object, "moving_sphere");
        return make_shared<moving_sphere>(
            read_vec3(object, "center0", "moving_sphere"),
            read_vec3(object, "center1", "moving_sphere"),
            read_double_or(object, "time0", 0.0),
            read_double_or(object, "time1", 1.0),
            read_double_or(object, "radius", 1.0), mat);
    }
    if (type == "box") {
        auto mat = lookup_material(materials, object, "box");
        return make_shared<box>(read_vec3(object, "min", "box"),
                                read_vec3(object, "max", "box"), mat);
    }
    if (type == "xy_rect") {
        auto mat = lookup_material(materials, object, "xy_rect");
        return make_shared<xy_rect>(
            read_double_or(object, "x0", 0.0), read_double_or(object, "x1", 1.0),
            read_double_or(object, "y0", 0.0), read_double_or(object, "y1", 1.0),
            read_double_or(object, "k", 0.0), mat);
    }
    if (type == "xz_rect") {
        auto mat = lookup_material(materials, object, "xz_rect");
        return make_shared<xz_rect>(
            read_double_or(object, "x0", 0.0), read_double_or(object, "x1", 1.0),
            read_double_or(object, "z0", 0.0), read_double_or(object, "z1", 1.0),
            read_double_or(object, "k", 0.0), mat);
    }
    if (type == "yz_rect") {
        auto mat = lookup_material(materials, object, "yz_rect");
        return make_shared<yz_rect>(
            read_double_or(object, "y0", 0.0), read_double_or(object, "y1", 1.0),
            read_double_or(object, "z0", 0.0), read_double_or(object, "z1", 1.0),
            read_double_or(object, "k", 0.0), mat);
    }
    if (type == "quad") {
        auto mat = lookup_material(materials, object, "quad");
        point3 q = read_vec3(object, "Q", "quad");
        vec3 u = read_vec3(object, "u", "quad");
        vec3 v = read_vec3(object, "v", "quad");
        auto quad = make_shared<hittable_list>();
        quad->add(make_shared<triangle>(q, q + u, q + u + v, mat));
        quad->add(make_shared<triangle>(q, q + u + v, q + v, mat));
        return quad;
    }
    if (type == "triangle") {
        auto mat = lookup_material(materials, object, "triangle");
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
        auto mat = lookup_material(materials, object, "obj");
        if (object.contains("implementation")) {
            throw std::runtime_error(
                "Scene file error: obj.implementation was removed; FlatMesh "
                "is now the only OBJ path.");
        }
        shared_ptr<hittable> mesh_object = FlatMesh::load_from_obj(
            read_string(object, "path", "obj"), mat,
            read_vec3_or(object, "translate", vec3(0, 0, 0), "obj"),
            read_vec3_or(object, "scale", vec3(1, 1, 1), "obj"),
            read_bool_or(object, "build_bvh", true),
            read_bool_or(object, "use_vertex_normals", true));
        if (!mesh_object) {
            throw std::runtime_error(
                "Scene file error: failed to load OBJ mesh.");
        }
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
        return make_shared<translate>(build_object(child, materials, textures),
                                      offset);
    }
    if (type == "rotate_y") {
        return make_shared<rotate_y>(
            build_object(require_key(object, "object", "rotate_y"), materials,
                         textures),
            read_double_or(object, "angle", 0.0));
    }
    if (type == "flip_face") {
        return make_shared<flip_face>(
            build_object(require_key(object, "object", "flip_face"),
                         materials, textures));
    }
    if (type == "constant_medium") {
        auto boundary =
            build_object(object.contains("boundary")
                             ? object["boundary"]
                             : require_key(object, "object",
                                           "constant_medium"),
                         materials, textures);
        double density = read_double_or(object, "density", 1.0);
        if (object.contains("texture")) {
            return make_shared<constant_medium>(
                boundary, density,
                build_texture_value(object["texture"], textures,
                                    "constant_medium.texture"));
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
            list.add(build_object(child, materials, textures));
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

void add_object(const json &object, const MaterialMap &materials,
                const TextureMap &textures,
                hittable_list &world) {
    world.add(build_object(object, materials, textures));
}


} // namespace scene_loader_internal

#include "scene_loader.h"

#include "accel.h"
#include "aarect.h"
#include "box.h"
#include "constant_medium.h"
#include "directional_light.h"
#include "environmental_light.h"
#include "json.hpp"
#include "material.h"
#include "mesh.h"
#include "moving_sphere.h"
#include "point_light.h"
#include "quad_light.h"
#include "sphere.h"
#include "spot_light.h"
#include "triangle.h"
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

using json = nlohmann::json;

namespace {

using MaterialMap = std::map<std::string, shared_ptr<material>>;
using TextureMap = std::map<std::string, shared_ptr<texture>>;

const json &require_key(const json &object, const std::string &key,
                        const std::string &context) {
    auto found = object.find(key);
    if (found == object.end()) {
        throw std::runtime_error("Scene file error: missing '" + key +
                                 "' in " + context + ".");
    }
    return *found;
}

vec3 read_vec3_value(const json &value, const std::string &context) {
    if (!value.is_array() || value.size() != 3) {
        throw std::runtime_error("Scene file error: expected 3-number array for " +
                                 context + ".");
    }
    return vec3(value[0].get<double>(), value[1].get<double>(),
                value[2].get<double>());
}

vec2 read_vec2_value(const json &value, const std::string &context) {
    if (!value.is_array() || value.size() != 2) {
        throw std::runtime_error("Scene file error: expected 2-number array for " +
                                 context + ".");
    }
    return vec2(value[0].get<double>(), value[1].get<double>());
}

vec3 read_vec3(const json &object, const std::string &key,
               const std::string &context) {
    return read_vec3_value(require_key(object, key, context), context + "." +
                                                               key);
}

vec3 read_vec3_or(const json &object, const std::string &key,
                  const vec3 &fallback, const std::string &context) {
    auto found = object.find(key);
    if (found == object.end()) {
        return fallback;
    }
    return read_vec3_value(*found, context + "." + key);
}

double read_double_or(const json &object, const std::string &key,
                      double fallback) {
    auto found = object.find(key);
    return found == object.end() ? fallback : found->get<double>();
}

int read_int_or(const json &object, const std::string &key, int fallback) {
    auto found = object.find(key);
    return found == object.end() ? fallback : found->get<int>();
}

bool read_bool_or(const json &object, const std::string &key, bool fallback) {
    auto found = object.find(key);
    return found == object.end() ? fallback : found->get<bool>();
}

std::string read_string(const json &object, const std::string &key,
                        const std::string &context) {
    return require_key(object, key, context).get<std::string>();
}

const json &material_field_or(const json &material_json, const std::string &a,
                              const std::string &b,
                              const std::string &context) {
    auto first = material_json.find(a);
    if (first != material_json.end()) {
        return *first;
    }
    return require_key(material_json, b, context);
}

shared_ptr<texture> lookup_texture(const TextureMap &textures,
                                   const std::string &name,
                                   const std::string &context) {
    auto found = textures.find(name);
    if (found == textures.end()) {
        throw std::runtime_error("Scene file error: unknown texture '" + name +
                                 "' in " + context + ".");
    }
    return found->second;
}

shared_ptr<texture> build_texture_value(const json &texture_json,
                                        const TextureMap &textures,
                                        const std::string &context);

shared_ptr<texture> build_texture_object(const json &texture_json,
                                         const TextureMap &textures,
                                         const std::string &context) {
    if (texture_json.contains("ref")) {
        return lookup_texture(textures, texture_json["ref"].get<std::string>(),
                              context);
    }

    std::string type = read_string(texture_json, "type", context);
    if (type == "solid") {
        if (texture_json.contains("color")) {
            return make_shared<solid_color>(
                read_vec3(texture_json, "color", context));
        }
        if (texture_json.contains("value")) {
            const auto &value = texture_json["value"];
            if (value.is_number()) {
                double scalar = value.get<double>();
                return make_shared<solid_color>(scalar, scalar, scalar);
            }
            return make_shared<solid_color>(
                read_vec3_value(value, context + ".value"));
        }
        throw std::runtime_error(
            "Scene file error: solid texture needs 'color' or 'value' in " +
            context + ".");
    }
    if (type == "checker") {
        const json &even =
            texture_json.contains("even") ? texture_json["even"]
                                          : require_key(texture_json, "color1",
                                                        context);
        const json &odd =
            texture_json.contains("odd") ? texture_json["odd"]
                                         : require_key(texture_json, "color2",
                                                       context);
        return make_shared<checker_texture>(
            build_texture_value(even, textures, context + ".even"),
            build_texture_value(odd, textures, context + ".odd"));
    }
    if (type == "noise") {
        return make_shared<noise_texture>(
            read_double_or(texture_json, "scale", 1.0));
    }
    if (type == "image") {
        return make_shared<image_texture>(
            read_string(texture_json, "path", context).c_str());
    }

    throw std::runtime_error("Scene file error: unknown texture type '" + type +
                             "' in " + context + ".");
}

shared_ptr<texture> build_texture_value(const json &texture_json,
                                        const TextureMap &textures,
                                        const std::string &context) {
    if (texture_json.is_string()) {
        return lookup_texture(textures, texture_json.get<std::string>(),
                              context);
    }
    if (texture_json.is_number()) {
        double scalar = texture_json.get<double>();
        return make_shared<solid_color>(scalar, scalar, scalar);
    }
    if (texture_json.is_array()) {
        return make_shared<solid_color>(
            read_vec3_value(texture_json, context));
    }
    if (texture_json.is_object()) {
        return build_texture_object(texture_json, textures, context);
    }
    throw std::runtime_error("Scene file error: invalid texture value in " +
                             context + ".");
}

shared_ptr<material>
build_material(const json &material_json, const TextureMap &textures,
               const std::string &name) {
    std::string type = read_string(material_json, "type", "material '" + name +
                                                        "'");

    if (type == "lambertian") {
        const auto &albedo =
            material_json.contains("texture")
                ? material_json["texture"]
                : material_field_or(material_json, "albedo", "color",
                                    "material '" + name + "'");
        return make_shared<lambertian>(
            build_texture_value(albedo, textures, "material '" + name + "'"));
    }
    if (type == "metal") {
        return make_shared<metal>(
            read_vec3(material_json, "albedo", "material '" + name + "'"),
            read_double_or(material_json, "fuzz", 0.0));
    }
    if (type == "dielectric") {
        double ir = read_double_or(
            material_json, "ir",
            read_double_or(material_json, "index_of_refraction", 1.5));
        return make_shared<dielectric>(ir);
    }
    if (type == "diffuse_light") {
        const auto &emit =
            material_json.contains("texture")
                ? material_json["texture"]
                : material_field_or(material_json, "emit", "color",
                                    "material '" + name + "'");
        return make_shared<diffuse_light>(
            build_texture_value(emit, textures, "material '" + name + "'"));
    }
    if (type == "pbr") {
        return make_shared<PBRMaterial>(
            build_texture_value(require_key(material_json, "albedo",
                                            "material '" + name + "'"),
                                textures, "material '" + name + "'.albedo"),
            build_texture_value(material_json.contains("roughness")
                                    ? material_json["roughness"]
                                    : json(0.5),
                                textures, "material '" + name + "'.roughness"),
            build_texture_value(material_json.contains("metallic")
                                    ? material_json["metallic"]
                                    : json(0.0),
                                textures, "material '" + name + "'.metallic"),
            material_json.contains("normal")
                ? build_texture_value(material_json["normal"], textures,
                                      "material '" + name + "'.normal")
                : nullptr);
    }

    throw std::runtime_error("Scene file error: unknown material type '" + type +
                             "' for material '" + name + "'.");
}

shared_ptr<material>
lookup_material(const MaterialMap &materials, const json &object,
                const std::string &context) {
    std::string material_name = read_string(object, "material", context);
    auto found = materials.find(material_name);
    if (found == materials.end()) {
        throw std::runtime_error("Scene file error: unknown material '" +
                                 material_name + "' in " + context + ".");
    }
    return found->second;
}

vec2 read_optional_uv(const json &object, const std::string &key,
                      bool &present, const std::string &context) {
    auto found = object.find(key);
    if (found == object.end()) {
        present = false;
        return vec2(0, 0);
    }
    present = true;
    return read_vec2_value(*found, context + "." + key);
}

shared_ptr<hittable> build_random_scene_generator(bool emissive_variant) {
    hittable_list world;

    auto checker = make_shared<checker_texture>(color(0.2, 0.3, 0.1),
                                                color(0.9, 0.9, 0.9));
    world.add(make_shared<sphere>(point3(0, -1000, 0), 1000,
                                  make_shared<lambertian>(checker)));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double();
            point3 center(a + 0.9 * random_double(), 0.2,
                          b + 0.9 * random_double());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                shared_ptr<material> sphere_material;

                if (!emissive_variant) {
                    if (choose_mat < 0.8) {
                        auto albedo = color::random() * color::random();
                        sphere_material = make_shared<lambertian>(albedo);
                        auto center2 =
                            center + vec3(0, random_double(0, .5), 0);
                        world.add(make_shared<moving_sphere>(
                            center, center2, 0.0, 1.0, 0.2,
                            sphere_material));
                    } else if (choose_mat < 0.95) {
                        auto albedo = color::random(0.5, 1);
                        auto fuzz = random_double(0, 0.5);
                        sphere_material = make_shared<metal>(albedo, fuzz);
                        world.add(make_shared<sphere>(center, 0.2,
                                                      sphere_material));
                    }
                } else {
                    if (choose_mat < 0.3) {
                        auto albedo = color::random() * color::random();
                        sphere_material = make_shared<lambertian>(albedo);
                        auto center2 =
                            center + vec3(0, random_double(0, .5), 0);
                        (void)center2;
                        world.add(make_shared<sphere>(center, 0.2,
                                                      sphere_material));
                    } else if (choose_mat < 0.6) {
                        auto albedo = color::random(0.5, 1);
                        auto fuzz = random_double(0, 0.5);
                        sphere_material = make_shared<metal>(albedo, fuzz);
                        world.add(make_shared<sphere>(center, 0.2,
                                                      sphere_material));
                    } else if (choose_mat < 0.95) {
                        auto difflight =
                            make_shared<diffuse_light>(color::random() * 2);
                        world.add(
                            make_shared<sphere>(center, 0.2, difflight));
                    }
                }
            }
        }
    }

    auto material1 = make_shared<dielectric>(1.5);
    world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

    if (emissive_variant) {
        auto material2 = make_shared<diffuse_light>(color(0.4, 0.2, 0.1) * 5);
        world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));
    } else {
        auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
        world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));
    }

    auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

    return make_accel(world, 0, 1);
}

shared_ptr<hittable> build_final_scene_generator(bool nee_variant) {
    hittable_list boxes1;
    auto ground = make_shared<lambertian>(color(0.48, 0.83, 0.53));

    const int boxes_per_side = 20;
    for (int i = 0; i < boxes_per_side; i++) {
        for (int j = 0; j < boxes_per_side; j++) {
            auto w = 100.0;
            auto x0 = -1000.0 + i * w;
            auto z0 = -1000.0 + j * w;
            auto y0 = 0.0;
            auto x1 = x0 + w;
            auto y1 = random_double(1, 101);
            auto z1 = z0 + w;

            boxes1.add(make_shared<box>(point3(x0, y0, z0),
                                        point3(x1, y1, z1), ground));
        }
    }

    hittable_list objects;
    objects.add(make_accel(boxes1, 0, 1));

    auto light = make_shared<diffuse_light>(color(7, 7, 7));
    auto light_rect = make_shared<xz_rect>(123, 423, 147, 412, 554, light);
    if (nee_variant) {
        objects.add(make_shared<flip_face>(light_rect));
    } else {
        objects.add(light_rect);
    }

    auto center1 = point3(400, 400, 200);
    auto center2 = center1 + vec3(30, 0, 0);
    auto moving_sphere_material = make_shared<lambertian>(color(0.7, 0.3, 0.1));
    objects.add(make_shared<moving_sphere>(center1, center2, 0, 1, 50,
                                           moving_sphere_material));

    objects.add(make_shared<sphere>(point3(260, 150, 45), 50,
                                    make_shared<dielectric>(1.5)));
    objects.add(
        make_shared<sphere>(point3(0, 150, 145), 50,
                            make_shared<metal>(color(0.8, 0.8, 0.9), 1.0)));

    auto boundary = make_shared<sphere>(point3(360, 150, 145), 70,
                                        make_shared<dielectric>(1.5));
    objects.add(boundary);
    objects.add(
        make_shared<constant_medium>(boundary, 0.2, color(0.2, 0.4, 0.9)));

    boundary = make_shared<sphere>(point3(0, 0, 0), 5000,
                                   make_shared<dielectric>(1.5));
    objects.add(make_shared<constant_medium>(boundary, .0001, color(1, 1, 1)));

    auto emat =
        make_shared<lambertian>(make_shared<image_texture>("earthmap.jpg"));
    objects.add(make_shared<sphere>(point3(400, 200, 400), 100, emat));

    auto pertext = make_shared<noise_texture>(0.1);
    objects.add(make_shared<sphere>(point3(220, 280, 300), 80,
                                    make_shared<lambertian>(pertext)));

    hittable_list boxes2;
    auto white = make_shared<lambertian>(color(.73, .73, .73));
    int ns = 1000;
    for (int j = 0; j < ns; j++) {
        boxes2.add(make_shared<sphere>(point3::random(0, 165), 10, white));
    }

    objects.add(make_shared<translate>(
        make_shared<rotate_y>(make_accel(boxes2, 0.0, 1.0), 15),
        vec3(-100, 270, 395)));

    return make_accel(objects, 0, 1);
}

shared_ptr<hittable> build_object(const json &object,
                                  const MaterialMap &materials,
                                  const TextureMap &textures) {
    std::string type = read_string(object, "type", "object");

    if (type == "random_scene_generator") {
        return build_random_scene_generator(
            read_bool_or(object, "emissive_variant", false));
    }
    if (type == "final_scene_generator") {
        return build_final_scene_generator(read_bool_or(object, "nee", false));
    }

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
        std::string implementation =
            object.contains("implementation")
                ? object["implementation"].get<std::string>()
                : "flat";
        shared_ptr<hittable> mesh_object;
        if (implementation == "legacy") {
            mesh_object = mesh::load_from_obj(
                read_string(object, "path", "obj"), mat,
                read_vec3_or(object, "translate", vec3(0, 0, 0), "obj"),
                read_vec3_or(object, "scale", vec3(1, 1, 1), "obj"),
                read_bool_or(object, "build_bvh", true),
                read_bool_or(object, "use_vertex_normals", true));
        } else if (implementation == "flat") {
            mesh_object = FlatMesh::load_from_obj(
                read_string(object, "path", "obj"), mat,
                read_vec3_or(object, "translate", vec3(0, 0, 0), "obj"),
                read_vec3_or(object, "scale", vec3(1, 1, 1), "obj"),
                read_bool_or(object, "build_bvh", true),
                read_bool_or(object, "use_vertex_normals", true));
        } else {
            throw std::runtime_error(
                "Scene file error: obj.implementation must be 'flat' or "
                "'legacy'.");
        }
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

void add_light(const json &light_json, SceneConfig &config) {
    std::string type = read_string(light_json, "type", "light");
    if (type == "point") {
        config.scene.lights.push_back(make_shared<PointLight>(
            read_vec3(light_json, "position", "point light"),
            read_vec3(light_json, "intensity", "point light")));
        return;
    }
    if (type == "directional") {
        config.scene.lights.push_back(make_shared<DirectionalLight>(
            read_vec3(light_json, "direction", "directional light"),
            read_vec3(light_json, "color", "directional light")));
        return;
    }
    if (type == "spot") {
        config.scene.lights.push_back(make_shared<SpotLight>(
            read_vec3(light_json, "position", "spot light"),
            read_vec3(light_json, "direction", "spot light"),
            read_double_or(light_json, "cutoff", 20.0),
            read_vec3(light_json, "intensity", "spot light")));
        return;
    }
    if (type == "quad") {
        config.scene.lights.push_back(make_shared<QuadLight>(
            read_vec3(light_json, "Q", "quad light"),
            read_vec3(light_json, "u", "quad light"),
            read_vec3(light_json, "v", "quad light"),
            read_vec3(light_json, "intensity", "quad light")));
        return;
    }
    if (type == "environment") {
        std::string path = read_string(light_json, "path", "environment light");
        config.scene.lights.push_back(
            make_shared<EnvironmentLight>(path.c_str()));
        return;
    }

    throw std::runtime_error("Scene file error: unknown light type '" + type +
                             "'.");
}

void apply_camera_json(const json &root, SceneConfig &config) {
    if (!root.contains("camera")) {
        return;
    }
    const auto &camera_json = root["camera"];
    config.camera.lookfrom =
        read_vec3_or(camera_json, "lookfrom", config.camera.lookfrom,
                     "camera");
    config.camera.lookat =
        read_vec3_or(camera_json, "lookat", config.camera.lookat, "camera");
    config.camera.vup =
        read_vec3_or(camera_json, "vup", config.camera.vup, "camera");
    config.camera.vfov =
        read_double_or(camera_json, "vfov", config.camera.vfov);
    config.camera.aperture =
        read_double_or(camera_json, "aperture", config.camera.aperture);
    config.camera.focus_dist =
        read_double_or(camera_json, "focus_dist", config.camera.focus_dist);
    config.camera.aspect_ratio =
        read_double_or(camera_json, "aspect_ratio",
                       config.camera.aspect_ratio);
}

void apply_render_json(const json &root, SceneConfig &config) {
    if (!root.contains("render")) {
        return;
    }
    const auto &render_json = root["render"];
    config.preset.image_width =
        read_int_or(render_json, "width", config.preset.image_width);
    config.preset.samples_per_pixel =
        read_int_or(render_json, "spp", config.preset.samples_per_pixel);
    config.preset.background =
        read_vec3_or(render_json, "background", config.preset.background,
                     "render");
}

} // namespace

SceneConfig load_scene_file(const std::string &path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Scene file error: cannot open '" + path +
                                 "'.");
    }

    json root;
    input >> root;

    SceneConfig config;

    apply_camera_json(root, config);
    apply_render_json(root, config);

    TextureMap textures;
    if (root.contains("textures")) {
        const auto &textures_json = root["textures"];
        if (!textures_json.is_object()) {
            throw std::runtime_error(
                "Scene file error: 'textures' must be an object.");
        }
        for (auto it = textures_json.begin(); it != textures_json.end(); ++it) {
            textures[it.key()] =
                build_texture_value(it.value(), textures,
                                    "texture '" + it.key() + "'");
        }
    }

    MaterialMap materials;
    const auto &materials_json = require_key(root, "materials", "root");
    for (auto it = materials_json.begin(); it != materials_json.end(); ++it) {
        materials[it.key()] = build_material(it.value(), textures, it.key());
    }

    hittable_list world;
    const auto &objects_json = require_key(root, "objects", "root");
    if (!objects_json.is_array()) {
        throw std::runtime_error("Scene file error: 'objects' must be an array.");
    }
    for (const auto &object : objects_json) {
        add_object(object, materials, textures, world);
    }

    if (root.contains("lights")) {
        const auto &lights_json = root["lights"];
        if (!lights_json.is_array()) {
            throw std::runtime_error(
                "Scene file error: 'lights' must be an array.");
        }
        for (const auto &light_json : lights_json) {
            add_light(light_json, config);
        }
    }

    if (read_bool_or(root, "world_accel", true)) {
        config.scene.world =
            make_accel(world, read_double_or(root, "time0", 0.0),
                       read_double_or(root, "time1", 1.0));
    } else {
        config.scene.world = make_shared<hittable_list>(world);
    }
    return config;
}

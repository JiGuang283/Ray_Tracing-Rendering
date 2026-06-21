#include "scene_loader.h"

#include "accel.h"
#include "aarect.h"
#include "box.h"
#include "directional_light.h"
#include "environmental_light.h"
#include "json.hpp"
#include "material.h"
#include "mesh.h"
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

shared_ptr<material>
build_material(const json &material_json, const std::string &name) {
    std::string type = read_string(material_json, "type", "material '" + name +
                                                        "'");

    if (type == "lambertian") {
        return make_shared<lambertian>(
            read_vec3(material_json, "albedo", "material '" + name + "'"));
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
        return make_shared<diffuse_light>(
            read_vec3(material_json, "color", "material '" + name + "'"));
    }
    if (type == "pbr") {
        color albedo =
            read_vec3(material_json, "albedo", "material '" + name + "'");
        double roughness = read_double_or(material_json, "roughness", 0.5);
        double metallic = read_double_or(material_json, "metallic", 0.0);
        return make_shared<PBRMaterial>(
            make_shared<solid_color>(albedo),
            make_shared<solid_color>(color(roughness, roughness, roughness)),
            make_shared<solid_color>(color(metallic, metallic, metallic)));
    }

    throw std::runtime_error("Scene file error: unknown material type '" + type +
                             "' for material '" + name + "'.");
}

shared_ptr<material>
lookup_material(const std::map<std::string, shared_ptr<material>> &materials,
                const json &object, const std::string &context) {
    std::string material_name = read_string(object, "material", context);
    auto found = materials.find(material_name);
    if (found == materials.end()) {
        throw std::runtime_error("Scene file error: unknown material '" +
                                 material_name + "' in " + context + ".");
    }
    return found->second;
}

void add_object(const json &object,
                const std::map<std::string, shared_ptr<material>> &materials,
                hittable_list &world) {
    std::string type = read_string(object, "type", "object");
    auto mat = lookup_material(materials, object, "object '" + type + "'");

    if (type == "sphere") {
        world.add(make_shared<sphere>(
            read_vec3(object, "center", "sphere"),
            read_double_or(object, "radius", 1.0), mat));
        return;
    }
    if (type == "box") {
        world.add(make_shared<box>(read_vec3(object, "min", "box"),
                                   read_vec3(object, "max", "box"), mat));
        return;
    }
    if (type == "xy_rect") {
        world.add(make_shared<xy_rect>(
            read_double_or(object, "x0", 0.0), read_double_or(object, "x1", 1.0),
            read_double_or(object, "y0", 0.0), read_double_or(object, "y1", 1.0),
            read_double_or(object, "k", 0.0), mat));
        return;
    }
    if (type == "xz_rect") {
        world.add(make_shared<xz_rect>(
            read_double_or(object, "x0", 0.0), read_double_or(object, "x1", 1.0),
            read_double_or(object, "z0", 0.0), read_double_or(object, "z1", 1.0),
            read_double_or(object, "k", 0.0), mat));
        return;
    }
    if (type == "yz_rect") {
        world.add(make_shared<yz_rect>(
            read_double_or(object, "y0", 0.0), read_double_or(object, "y1", 1.0),
            read_double_or(object, "z0", 0.0), read_double_or(object, "z1", 1.0),
            read_double_or(object, "k", 0.0), mat));
        return;
    }
    if (type == "quad") {
        point3 q = read_vec3(object, "Q", "quad");
        vec3 u = read_vec3(object, "u", "quad");
        vec3 v = read_vec3(object, "v", "quad");
        world.add(make_shared<triangle>(q, q + u, q + u + v, mat));
        world.add(make_shared<triangle>(q, q + u + v, q + v, mat));
        return;
    }
    if (type == "obj") {
        auto mesh_object = FlatMesh::load_from_obj(
            read_string(object, "path", "obj"), mat,
            read_vec3_or(object, "translate", vec3(0, 0, 0), "obj"),
            read_vec3_or(object, "scale", vec3(1, 1, 1), "obj"),
            read_bool_or(object, "build_bvh", true),
            read_bool_or(object, "use_vertex_normals", true));
        if (!mesh_object) {
            throw std::runtime_error(
                "Scene file error: failed to load OBJ mesh.");
        }
        world.add(mesh_object);
        return;
    }

    throw std::runtime_error("Scene file error: unknown object type '" + type +
                             "'.");
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

    if (root.contains("camera")) {
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
        config.camera.aspect_ratio = read_double_or(
            camera_json, "aspect_ratio", config.camera.aspect_ratio);
    }

    if (root.contains("render")) {
        const auto &render_json = root["render"];
        config.preset.image_width =
            read_int_or(render_json, "width", config.preset.image_width);
        config.preset.samples_per_pixel =
            read_int_or(render_json, "spp", config.preset.samples_per_pixel);
        config.preset.background =
            read_vec3_or(render_json, "background", config.preset.background,
                         "render");
    }

    std::map<std::string, shared_ptr<material>> materials;
    const auto &materials_json = require_key(root, "materials", "root");
    for (auto it = materials_json.begin(); it != materials_json.end(); ++it) {
        materials[it.key()] = build_material(it.value(), it.key());
    }

    hittable_list world;
    const auto &objects_json = require_key(root, "objects", "root");
    if (!objects_json.is_array()) {
        throw std::runtime_error("Scene file error: 'objects' must be an array.");
    }
    for (const auto &object : objects_json) {
        add_object(object, materials, world);
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

    config.scene.world = make_accel(world, 0.0, 1.0);
    return config;
}

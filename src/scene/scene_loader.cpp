#include "scene_loader.h"

#include "accel.h"
#include "hittable_list.h"
#include "json.hpp"
#include "scene_loader_internal.h"

#include <fstream>
#include <stdexcept>

using json = nlohmann::json;

namespace {

using namespace scene_loader_internal;

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

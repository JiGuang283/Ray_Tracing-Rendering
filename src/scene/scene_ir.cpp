#include "scene_ir.h"

#include "scene_loader_internal.h"

#include <stdexcept>

using json = nlohmann::json;

namespace {

using namespace scene_loader_internal;

SceneNamedSpec parse_named_spec(const std::string &name, const json &data,
                                const std::string &context) {
    if (!data.is_object()) {
        throw std::runtime_error("Scene file error: " + context +
                                 " must be an object.");
    }
    SceneNamedSpec spec;
    spec.name = name;
    spec.type = read_string(data, "type", context);
    spec.data = data;
    return spec;
}

SceneObjectSpec parse_object_spec(const json &data,
                                  const std::string &context) {
    if (!data.is_object()) {
        throw std::runtime_error("Scene file error: " + context +
                                 " must be an object.");
    }
    SceneObjectSpec spec;
    spec.type = read_string(data, "type", context);
    spec.data = data;
    return spec;
}

ToneMappingMode parse_tone_mapping(const std::string &mode,
                                   const std::string &context) {
    if (mode == "linear") {
        return ToneMappingMode::Linear;
    }
    if (mode == "reinhard") {
        return ToneMappingMode::Reinhard;
    }
    if (mode == "aces" || mode == "ACES") {
        return ToneMappingMode::ACES;
    }
    throw std::runtime_error("Scene file error: unknown tone_mapping '" +
                             mode + "' in " + context + ".");
}

void apply_color_pipeline_json(const json &pipeline_json,
                               ColorPipelineSettings &settings,
                               const std::string &context) {
    settings.exposure =
        read_double_or(pipeline_json, "exposure", settings.exposure);
    settings.gamma = read_double_or(pipeline_json, "gamma", settings.gamma);
    if (pipeline_json.contains("tone_mapping")) {
        settings.tone_mapping = parse_tone_mapping(
            pipeline_json["tone_mapping"].get<std::string>(),
            context + ".tone_mapping");
    }
}

void apply_camera_json(const json &root, SceneIR &ir) {
    if (!root.contains("camera")) {
        return;
    }
    const auto &camera_json = root["camera"];
    ir.camera.lookfrom =
        read_vec3_or(camera_json, "lookfrom", ir.camera.lookfrom, "camera");
    ir.camera.lookat =
        read_vec3_or(camera_json, "lookat", ir.camera.lookat, "camera");
    ir.camera.vup = read_vec3_or(camera_json, "vup", ir.camera.vup, "camera");
    ir.camera.vfov = read_double_or(camera_json, "vfov", ir.camera.vfov);
    ir.camera.aperture =
        read_double_or(camera_json, "aperture", ir.camera.aperture);
    ir.camera.focus_dist =
        read_double_or(camera_json, "focus_dist", ir.camera.focus_dist);
    ir.camera.aspect_ratio = read_double_or(
        camera_json, "aspect_ratio", ir.camera.aspect_ratio);
}

void apply_render_json(const json &root, SceneIR &ir) {
    if (!root.contains("render")) {
        return;
    }
    const auto &render_json = root["render"];
    ir.preset.image_width =
        read_int_or(render_json, "width", ir.preset.image_width);
    ir.preset.samples_per_pixel =
        read_int_or(render_json, "spp", ir.preset.samples_per_pixel);
    ir.preset.background =
        read_vec3_or(render_json, "background", ir.preset.background,
                     "render");
    apply_color_pipeline_json(render_json, ir.preset.color_pipeline,
                              "render");
    if (render_json.contains("color_pipeline")) {
        const auto &pipeline_json = render_json["color_pipeline"];
        if (!pipeline_json.is_object()) {
            throw std::runtime_error(
                "Scene file error: render.color_pipeline must be an object.");
        }
        apply_color_pipeline_json(pipeline_json, ir.preset.color_pipeline,
                                  "render.color_pipeline");
    }
}

} // namespace

SceneIR parse_scene_ir(const SceneDescription &description) {
    const json &root = description.root;
    if (!root.is_object()) {
        throw std::runtime_error("Scene file error: root must be an object.");
    }

    SceneIR ir;
    ir.source_path = description.source_path;
    if (root.contains("name")) {
        ir.name = root["name"].get<std::string>();
    }

    apply_camera_json(root, ir);
    apply_render_json(root, ir);

    ir.world_accel = read_bool_or(root, "world_accel", true);
    ir.time0 = read_double_or(root, "time0", 0.0);
    ir.time1 = read_double_or(root, "time1", 1.0);

    if (root.contains("textures")) {
        const auto &textures_json = root["textures"];
        if (!textures_json.is_object()) {
            throw std::runtime_error(
                "Scene file error: 'textures' must be an object.");
        }
        for (auto it = textures_json.begin(); it != textures_json.end(); ++it) {
            ir.textures.push_back(parse_named_spec(
                it.key(), it.value(), "texture '" + it.key() + "'"));
        }
    }

    const auto &materials_json = require_key(root, "materials", "root");
    if (!materials_json.is_object()) {
        throw std::runtime_error(
            "Scene file error: 'materials' must be an object.");
    }
    for (auto it = materials_json.begin(); it != materials_json.end(); ++it) {
        ir.materials.push_back(parse_named_spec(
            it.key(), it.value(), "material '" + it.key() + "'"));
    }

    const auto &objects_json = require_key(root, "objects", "root");
    if (!objects_json.is_array()) {
        throw std::runtime_error("Scene file error: 'objects' must be an array.");
    }
    for (size_t i = 0; i < objects_json.size(); ++i) {
        ir.objects.push_back(parse_object_spec(
            objects_json[i], "objects[" + std::to_string(i) + "]"));
    }

    bool has_explicit_lights = root.contains("lights") &&
                               root["lights"].is_array() &&
                               !root["lights"].empty();
    ir.auto_emitters =
        read_bool_or(root, "auto_emitters", !has_explicit_lights);

    if (root.contains("lights")) {
        const auto &lights_json = root["lights"];
        if (!lights_json.is_array()) {
            throw std::runtime_error(
                "Scene file error: 'lights' must be an array.");
        }
        for (size_t i = 0; i < lights_json.size(); ++i) {
            ir.lights.push_back(parse_object_spec(
                lights_json[i], "lights[" + std::to_string(i) + "]"));
        }
    }

    return ir;
}

#include "scene_loader.h"

#include "accel.h"
#include "hittable_list.h"
#include "json.hpp"
#include "scene_ir.h"
#include "scene_loader_internal.h"

#include <fstream>
#include <stdexcept>
#include <vector>

using json = nlohmann::json;

using namespace scene_loader_internal;

SceneDescription load_scene_description(const std::string &path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Scene file error: cannot open '" + path +
                                 "'.");
    }

    SceneDescription description;
    description.source_path = path;
    input >> description.root;
    return description;
}

SceneConfig build_scene_config(const SceneIR &ir) {
    SceneConfig config;
    config.camera = ir.camera;
    config.preset = ir.preset;

    SceneBuildContext context;
    context.source_path = ir.source_path;
    context.scene_ir = &ir;
    for (const auto &material : ir.materials) {
        context.materials[material.name] =
            build_material(material, context);
    }

    std::vector<shared_ptr<Light>> auto_lights;

    hittable_list world;
    for (ObjectIRId object : ir.objects) {
        add_object(object, context, world, auto_lights, ir.auto_emitters);
    }

    for (const auto &light : ir.lights) {
        add_light(light, context, config);
    }
    config.scene.lights.insert(config.scene.lights.end(), auto_lights.begin(),
                               auto_lights.end());

    if (ir.world_accel) {
        config.scene.world = make_accel(world, ir.time0, ir.time1);
    } else {
        config.scene.world = make_shared<hittable_list>(world);
    }
    return config;
}

SceneConfig build_scene_config(const SceneDescription &description) {
    return build_scene_config(parse_scene_ir(description));
}

SceneConfig load_scene_file(const std::string &path) {
    return build_scene_config(load_scene_description(path));
}

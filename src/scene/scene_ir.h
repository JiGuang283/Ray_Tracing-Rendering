#ifndef SCENE_IR_H
#define SCENE_IR_H

#include "json.hpp"
#include "scene_config.h"
#include "scene_description.h"

#include <string>
#include <vector>

struct SceneNamedSpec {
    std::string name;
    std::string type;
    nlohmann::json data;
};

struct SceneObjectSpec {
    std::string type;
    nlohmann::json data;
};

struct SceneIR {
    std::string source_path;
    std::string name;
    CameraConfig camera;
    RenderPreset preset;
    std::vector<SceneNamedSpec> textures;
    std::vector<SceneNamedSpec> materials;
    std::vector<SceneObjectSpec> objects;
    std::vector<SceneObjectSpec> lights;
    bool world_accel = true;
    bool auto_emitters = true;
    double time0 = 0.0;
    double time1 = 1.0;
};

SceneIR parse_scene_ir(const SceneDescription &description);
SceneConfig build_scene_config(const SceneIR &ir);

#endif

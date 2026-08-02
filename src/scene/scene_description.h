#ifndef SCENE_DESCRIPTION_H
#define SCENE_DESCRIPTION_H

#include "json.hpp"

#include <string>

struct SceneDescription {
    std::string source_path;
    nlohmann::json root;
};

SceneDescription load_scene_description(const std::string &path);

#endif

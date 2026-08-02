#ifndef SCENE_LOADER_H
#define SCENE_LOADER_H

#include "scene_config.h"
#include "scene_description.h"
#include "scene_ir.h"

#include <string>

SceneConfig build_scene_config(const SceneIR &ir);
SceneConfig build_scene_config(const SceneDescription &description);
SceneConfig load_scene_file(const std::string &path);

#endif

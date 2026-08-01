#ifndef SCENE_LOADER_H
#define SCENE_LOADER_H

#include "scene_description.h"
#include "scene_ir.h"
#include <string>

SceneIR load_scene_ir_file(const std::string &path);
SceneConfig load_scene_file(const std::string &path);

#endif

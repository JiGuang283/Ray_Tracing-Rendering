#ifndef SCENE_LOADER_H
#define SCENE_LOADER_H

#include "scene_description.h"
#include <string>

SceneConfig load_scene_file(const std::string &path);

#endif

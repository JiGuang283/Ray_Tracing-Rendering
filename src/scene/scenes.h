#ifndef SCENES_H
#define SCENES_H

#include "scene_config.h"

#include <string>

SceneConfig select_scene(int scene_id);
std::string scene_path(int scene_id);

#endif

#ifndef SCENE_CONFIG_H
#define SCENE_CONFIG_H

#include "hittable.h"
#include "light.h"
#include "scene_types.h"
#include <memory>
#include <vector>

using std::shared_ptr;

struct Scene {
    shared_ptr<hittable> world;
    std::vector<shared_ptr<Light>> lights;
};

struct SceneConfig {
    Scene scene;
    CameraConfig camera;
    RenderPreset preset;
};

#endif

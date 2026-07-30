#ifndef SCENE_CONFIG_H
#define SCENE_CONFIG_H

#include "hittable.h"
#include "light.h"
#include "color_pipeline.h"
#include "vec3.h"
#include <memory>
#include <vector>

using std::shared_ptr;

struct Scene {
    shared_ptr<hittable> world;
    std::vector<shared_ptr<Light>> lights;
};

struct CameraConfig {
    point3 lookfrom{13, 2, 3};
    point3 lookat{0, 0, 0};
    vec3 vup{0, 1, 0};
    double vfov = 40.0;
    double aperture = 0.0;
    double focus_dist = 10.0;
    double aspect_ratio = 16.0 / 9.0;
};

struct RenderPreset {
    int image_width = 1280;
    int samples_per_pixel = 100;
    color background{0, 0, 0};
    ColorPipelineSettings color_pipeline;
};

struct SceneConfig {
    Scene scene;
    CameraConfig camera;
    RenderPreset preset;
};

#endif

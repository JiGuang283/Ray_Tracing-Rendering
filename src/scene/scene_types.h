#ifndef SCENE_TYPES_H
#define SCENE_TYPES_H

#include "color_pipeline_settings.h"
#include "vec3.h"

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
    double sample_clamp = 0.0;
    ColorPipelineSettings color_pipeline;
};

#endif

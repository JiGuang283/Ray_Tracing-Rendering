#ifndef SCENE_TYPES_H
#define SCENE_TYPES_H

#include "camera_config.h"
#include "color_pipeline_settings.h"

struct RenderPreset {
    int image_width = 1280;
    int samples_per_pixel = 100;
    color background{0, 0, 0};
    double sample_clamp = 0.0;
    ColorPipelineSettings color_pipeline;
};

#endif

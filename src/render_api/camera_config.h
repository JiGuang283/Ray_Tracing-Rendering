#ifndef CAMERA_CONFIG_H
#define CAMERA_CONFIG_H

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

void validate_camera_config(const CameraConfig &camera);

#endif

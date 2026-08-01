#ifndef SAMPLE_FILTER_H
#define SAMPLE_FILTER_H

#include "vec3.h"

struct FilteredCameraSample {
    color radiance{0, 0, 0};
    bool clamped = false;
    bool invalid = false;
};

FilteredCameraSample filter_camera_sample(const color &radiance,
                                          double luminance_limit);

#endif

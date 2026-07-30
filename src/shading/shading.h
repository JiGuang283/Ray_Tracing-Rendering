#ifndef SHADING_H
#define SHADING_H

#include "shading/bsdf.h"

struct ShadingResult {
    BSDF bsdf;
    color emission;
    bool has_emission;

    ShadingResult() : bsdf(), emission(0, 0, 0), has_emission(false) {
    }

    void reset(const ShadingFrame &frame) {
        bsdf.reset(frame);
        emission = color(0, 0, 0);
        has_emission = false;
    }

    void set_emission(const color &value) {
        emission = value;
        has_emission = value.length_squared() > 0.0;
    }
};

#endif

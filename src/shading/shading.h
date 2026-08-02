#ifndef SHADING_H
#define SHADING_H

#include "bsdf.h"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <variant>

struct MaterialOutput {
    BSDF bsdf;
    color emission{0, 0, 0};
    double opacity = 1.0;
    bool has_emission = false;

    void reset(const ShadingFrame &frame) {
        bsdf.reset(frame);
        emission = color(0, 0, 0);
        opacity = 1.0;
        has_emission = false;
    }

    void reset(const ShadingFrame &frame, const vec3 &geometry_normal) {
        bsdf.reset(frame, geometry_normal);
        emission = color(0, 0, 0);
        opacity = 1.0;
        has_emission = false;
    }

    void set_emission(const color &value) {
        emission = value;
        has_emission = value.length_squared() > 0.0;
    }
};

using ShaderValue = std::variant<std::monostate, double, vec3>;

struct ShaderScratch {
    static constexpr std::size_t kCapacity = 64;
    std::array<ShaderValue, kCapacity> values;
    std::size_t used = 0;

    void reset() {
        used = 0;
    }

    ShaderValue &allocate() {
        if (used >= kCapacity) {
            throw std::overflow_error("ShaderScratch capacity exceeded");
        }
        return values[used++];
    }
};

#endif

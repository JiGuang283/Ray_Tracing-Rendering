#ifndef BSDF_CLOSURES_H
#define BSDF_CLOSURES_H

#include "interaction.h"
#include "rtweekend.h"

#include <optional>
#include <variant>

struct LambertianClosure {
    color albedo{0, 0, 0};
};

struct SpecularReflectionClosure {
    color reflectance{1, 1, 1};
};

struct SpecularDielectricClosure {
    double ior = 1.5;
    bool front_face = true;
};

struct GGXReflectionClosure {
    color f0{0.04, 0.04, 0.04};
    double roughness = 0.5;
};

struct ClearcoatGGXClosure {
    double roughness = 0.1;
};

struct IsotropicPhaseClosure {
    color albedo{1, 1, 1};
};

using ClosureVariant =
    std::variant<LambertianClosure, SpecularReflectionClosure,
                 SpecularDielectricClosure, GGXReflectionClosure,
                 ClearcoatGGXClosure, IsotropicPhaseClosure>;

namespace bsdf_closures {

double luminance(const color &value);
bool is_delta(const ClosureVariant &closure);
bool is_phase(const ClosureVariant &closure);

std::optional<BSDFSample> sample(const ClosureVariant &closure,
                                 const ShadingFrame &frame, const vec3 &wo,
                                 RNG &rng);
color eval(const ClosureVariant &closure, const ShadingFrame &frame,
           const vec3 &wo, const vec3 &wi);
double pdf(const ClosureVariant &closure, const ShadingFrame &frame,
           const vec3 &wo, const vec3 &wi);

} // namespace bsdf_closures

#endif

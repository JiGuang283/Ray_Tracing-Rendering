#ifndef BSDF_CLOSURES_H
#define BSDF_CLOSURES_H

#include "shading/bsdf.h"

namespace bsdf_closures {

double luminance(const color &c);

bool sample(const BSDFClosure &closure, const ShadingFrame &frame,
            const vec3 &wo, BSDFSample &sampled, RNG &rng);
color eval(const BSDFClosure &closure, const ShadingFrame &frame,
           const vec3 &wo, const vec3 &wi);
double pdf(const BSDFClosure &closure, const ShadingFrame &frame,
           const vec3 &wo, const vec3 &wi);

} // namespace bsdf_closures

#endif

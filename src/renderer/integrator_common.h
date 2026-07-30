#ifndef INTEGRATOR_COMMON_H
#define INTEGRATOR_COMMON_H

#include "hittable.h"
#include "light_sampler.h"
#include "material.h"
#include "shading/shading.h"

namespace integrator_common {

struct ShadedSurface {
    SurfaceInteraction surface;
    ShadingResult shading;
    vec3 wo;
    double time = 0.0;
};

ShadedSurface shade_surface(const hit_record &rec, const ray &r);
double power_heuristic(double pdf_a, double pdf_b);
color clamp_radiance(const color &L, double max_value = 100.0);
double scattering_cos_factor(const ShadingResult &shading,
                             const BSDFSample &sample);
color scattering_weight(const ShadingResult &shading,
                        const BSDFSample &sample);
bool visible(const hittable &scene, const ray &shadow_ray,
             double max_distance, RNG &rng);
color sample_direct_lighting(const ShadedSurface &shaded,
                             const hittable &scene,
                             const LightSampler &light_sampler, RNG &rng,
                             bool use_mis);

} // namespace integrator_common

#endif

#ifndef INTEGRATOR_COMMON_H
#define INTEGRATOR_COMMON_H

#include "hittable.h"
#include "light_sampler.h"
#include "material.h"
#include "shading.h"

namespace integrator_common {

struct ShadedSurface {
    SurfaceInteraction surface;
    MaterialOutput shading;
    vec3 wo;
    double time = 0.0;
};

ShadedSurface shade_surface(const hit_record &rec, const ray &r,
                            ShaderScratch &scratch);
double power_heuristic(double pdf_a, double pdf_b);
double scattering_cos_factor(const MaterialOutput &shading,
                             const BSDFSample &sample);
color scattering_weight(const MaterialOutput &shading,
                        const BSDFSample &sample);
bool visible(const hittable &scene, const ray &shadow_ray,
             double max_distance, RNG &rng);
color sample_direct_lighting(const ShadedSurface &shaded,
                             const hittable &scene,
                             const LightSampler &light_sampler, RNG &rng,
                             bool use_mis);

} // namespace integrator_common

#endif

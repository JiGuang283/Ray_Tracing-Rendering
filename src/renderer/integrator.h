#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include "hittable.h"
#include "light.h"
#include "light_sampler.h"
#include "ray.h"
#include "shading.h"
#include "vec3.h"

#include <cstdint>

struct IntegratorContext {
    RNG &rng;
    ShaderScratch &shader_scratch;
    const LightSampler *light_sampler = nullptr;
    // Optional per-worker counters for backend-neutral statistics.
    std::uint64_t *traversal_steps = nullptr;
    std::uint64_t *shadow_rays = nullptr;
};

class Integrator {
  public:
    virtual ~Integrator() = default;

    virtual color Li(const ray &r, const hittable &scene,
                     const color &background,
                     IntegratorContext &context) const = 0;

    virtual void set_max_depth(int depth) = 0;
};

#endif

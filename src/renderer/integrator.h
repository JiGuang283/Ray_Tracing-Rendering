#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include "hittable.h"
#include "light.h"
#include "light_sampler.h"
#include "ray.h"
#include "shading.h"
#include "vec3.h"

struct IntegratorContext {
    RNG &rng;
    ShaderScratch &shader_scratch;
    const LightSampler *light_sampler = nullptr;
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

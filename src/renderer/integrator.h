#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include "hittable.h"
#include "light.h"
#include "ray.h"
#include "shading/shading.h"
#include "vec3.h"

struct IntegratorContext {
    RNG &rng;
    ShaderScratch &shader_scratch;
};

class Integrator {
  public:
    virtual ~Integrator() = default;

    color Li(const ray &r, const hittable &scene,
             const color &background) const {
        RNG rng(make_thread_seed());
        ShaderScratch scratch;
        IntegratorContext context{rng, scratch};
        return Li(r, scene, background, context);
    }

    color Li(const ray &r, const hittable &scene, const color &background,
             const std::vector<shared_ptr<Light>> &lights) const {
        RNG rng(make_thread_seed());
        ShaderScratch scratch;
        IntegratorContext context{rng, scratch};
        return Li(r, scene, background, lights, context);
    }

    color Li(const ray &r, const hittable &scene, const color &background,
             RNG &rng) const {
        ShaderScratch scratch;
        IntegratorContext context{rng, scratch};
        return Li(r, scene, background, context);
    }

    color Li(const ray &r, const hittable &scene, const color &background,
             const std::vector<shared_ptr<Light>> &lights, RNG &rng) const {
        ShaderScratch scratch;
        IntegratorContext context{rng, scratch};
        return Li(r, scene, background, lights, context);
    }

    virtual color Li(const ray &r, const hittable &scene,
                     const color &background,
                     IntegratorContext &context) const = 0;

    virtual color
    Li(const ray &r, const hittable &scene, const color &background,
       const std::vector<shared_ptr<Light>> &lights,
       IntegratorContext &context) const {
        return Li(r, scene, background, context);
    }

    virtual void set_max_depth(int depth) = 0;
};

#endif

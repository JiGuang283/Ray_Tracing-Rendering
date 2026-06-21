#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include "hittable.h"
#include "light.h"
#include "ray.h"
#include "vec3.h"

class Integrator {
  public:
    virtual ~Integrator() = default;
    virtual color Li(const ray &r, const hittable &scene,
                     const color &background) const {
        RNG rng(make_thread_seed());
        return Li(r, scene, background, rng);
    }
    virtual color Li(const ray &r, const hittable &scene,
                     const color &background,
                     const std::vector<shared_ptr<Light>> &lights) const {
        RNG rng(make_thread_seed());
        return Li(r, scene, background, lights, rng);
    }
    virtual color Li(const ray &r, const hittable &scene,
                     const color &background, RNG &rng) const = 0;
    virtual color Li(const ray &r, const hittable &scene,
                     const color &background,
                     const std::vector<shared_ptr<Light>> &lights,
                     RNG &rng) const {
        return Li(r, scene, background, rng);
    }
    virtual void set_max_depth(int depth) = 0;
};

#endif

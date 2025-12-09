#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include "hittable.h"
#include "ray.h"
#include "vec3.h"

class Integrator {
  public:
    virtual ~Integrator() = default;
    virtual color Li(const ray &r, const hittable &scene,
                     const color &background) const = 0;
    virtual void set_max_depth(int depth) = 0;
};

#endif
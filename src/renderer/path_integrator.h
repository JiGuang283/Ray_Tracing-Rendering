#ifndef PATH_INTEGRATOR_H
#define PATH_INTEGRATOR_H

#include "integrator.h"

class PathIntegrator : public Integrator {
  public:
    PathIntegrator();

    void set_max_depth(int depth = 50) override;

    color Li(const ray &r, const hittable &scene,
             const color &background) const override;

    color Li(const ray &r, const hittable &scene, const color &background,
             RNG &rng) const override;

  private:
    color Li_internal(const ray &r, const hittable &scene,
                      const color &background, int depth, RNG &rng) const;

    int m_max_depth = 50;
};

#endif

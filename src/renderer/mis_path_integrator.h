#ifndef MIS_PATH_INTEGRATOR_H
#define MIS_PATH_INTEGRATOR_H

#include "integrator.h"
#include <vector>

class MISPathIntegrator : public Integrator {
  public:
    using Integrator::Li;

    MISPathIntegrator();

    void set_max_depth(int depth = 50) override;
    void set_rr_start_depth(int depth);

    color Li(const ray &r, const hittable &scene, const color &background,
             IntegratorContext &context) const override;
    color
    Li(const ray &r, const hittable &scene, const color &background,
       const std::vector<shared_ptr<Light>> &lights,
       IntegratorContext &context) const override;

  private:
    int m_max_depth = 50;
    int m_rr_start_depth = 3;
};

#endif

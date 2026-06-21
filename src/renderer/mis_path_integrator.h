#ifndef MIS_PATH_INTEGRATOR_H
#define MIS_PATH_INTEGRATOR_H

#include "integrator.h"
#include <vector>

class MISPathIntegrator : public Integrator {
  public:
    MISPathIntegrator();

    void set_max_depth(int depth = 50) override;
    void set_rr_start_depth(int depth);

    color Li(const ray &r, const hittable &scene,
             const color &background) const override;
    color
    Li(const ray &r, const hittable &scene, const color &background,
       const std::vector<shared_ptr<Light>> &lights) const override;
    color Li(const ray &r, const hittable &scene, const color &background,
             RNG &rng) const override;
    color
    Li(const ray &r, const hittable &scene, const color &background,
       const std::vector<shared_ptr<Light>> &lights,
       RNG &rng) const override;

  private:
    static color clamp_radiance(const color &L, double max_value = 100.0);
    static double power_heuristic(double pdf_a, double pdf_b);

    double compute_light_pdf(const hit_record &rec, const vec3 &wo,
                             const std::vector<shared_ptr<Light>> &lights,
                             const ray &current_ray) const;
    color
    sample_lights_mis(const hit_record &rec, const vec3 &wo,
                      const hittable &scene,
                      const std::vector<shared_ptr<Light>> &lights,
                      RNG &rng) const;

    int m_max_depth = 50;
    int m_rr_start_depth = 3;
};

#endif

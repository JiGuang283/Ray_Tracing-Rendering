#ifndef LIGHT_SAMPLER_H
#define LIGHT_SAMPLER_H

#include "light.h"
#include "rtweekend.h"

#include <vector>

class LightSampler {
  public:
    explicit LightSampler(const std::vector<shared_ptr<Light>> &lights);

    bool empty() const;
    LightSample sample(const point3 &p, RNG &rng,
                       double &light_selection_pdf) const;
    double pdf(const point3 &origin, const vec3 &direction) const;
    color environment_radiance(const ray &r, bool &found_environment) const;

  private:
    int choose_light(double u) const;
    double selection_pdf(int light_index) const;

    const std::vector<shared_ptr<Light>> &m_lights;
    std::vector<double> m_cdf;
    double m_total_weight = 0.0;
};

#endif

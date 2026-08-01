#ifndef LIGHT_SAMPLER_H
#define LIGHT_SAMPLER_H

#include "light.h"
#include "rtweekend.h"

#include <vector>

struct SelectedLightSample {
    LightSample sample;
    double selection_pdf = 0.0;
    bool has_bsdf_competitor = false;
};

class LightSampler {
  public:
    explicit LightSampler(const std::vector<shared_ptr<Light>> &lights);

    bool empty() const;
    bool has_non_delta_lights() const;
    const std::vector<shared_ptr<Light>> &delta_lights() const;
    SelectedLightSample sample_non_delta(const point3 &p, RNG &rng) const;
    double pdf(const point3 &origin, const vec3 &direction) const;
    color environment_radiance(const ray &r, bool &found_environment) const;

    std::size_t non_delta_light_count() const;
    double non_delta_selection_pdf(std::size_t light_index) const;

  private:
    int choose_light(double u) const;

    std::vector<shared_ptr<Light>> m_delta_lights;
    std::vector<shared_ptr<Light>> m_non_delta_lights;
    std::vector<double> m_selection_probabilities;
    std::vector<double> m_cdf;
};

#endif

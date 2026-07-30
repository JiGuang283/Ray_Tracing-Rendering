#include "light_sampler.h"

#include <algorithm>

LightSampler::LightSampler(const std::vector<shared_ptr<Light>> &lights)
    : m_lights(lights) {
    m_cdf.reserve(m_lights.size());
    double accum = 0.0;
    for (const auto &light : m_lights) {
        color p = light->power();
        double weight = std::max(0.0, p.x() + p.y() + p.z());
        accum += weight;
        m_cdf.push_back(accum);
    }
    m_total_weight = accum;
}

bool LightSampler::empty() const {
    return m_lights.empty();
}

LightSample LightSampler::sample(const point3 &p, RNG &rng,
                                 double &light_selection_pdf) const {
    LightSample result;
    result.Li = color(0, 0, 0);
    result.wi = vec3(0, 0, 1);
    result.pdf = 0.0;
    result.dist = infinity;
    result.is_delta = false;
    light_selection_pdf = 0.0;

    if (m_lights.empty()) {
        return result;
    }

    int light_index = choose_light(rng.next());
    light_selection_pdf = selection_pdf(light_index);
    return m_lights[light_index]->sample(p, vec2(rng.next(), rng.next()));
}

double LightSampler::pdf(const point3 &origin, const vec3 &direction) const {
    if (m_lights.empty()) {
        return 0.0;
    }

    double total_pdf = 0.0;
    for (int i = 0; i < static_cast<int>(m_lights.size()); ++i) {
        total_pdf += m_lights[i]->pdf(origin, direction) * selection_pdf(i);
    }
    return total_pdf;
}

color LightSampler::environment_radiance(
    const ray &r, bool &found_environment) const {
    color result(0, 0, 0);
    found_environment = false;
    for (const auto &light : m_lights) {
        if (light->is_infinite()) {
            result += light->Le(r);
            found_environment = true;
        }
    }
    return result;
}

int LightSampler::choose_light(double u) const {
    if (m_lights.empty()) {
        return 0;
    }
    if (m_total_weight <= 0.0) {
        int index = static_cast<int>(u * m_lights.size());
        return std::min(index, static_cast<int>(m_lights.size()) - 1);
    }

    double target = u * m_total_weight;
    auto it = std::lower_bound(m_cdf.begin(), m_cdf.end(), target);
    int index = static_cast<int>(it - m_cdf.begin());
    return std::min(index, static_cast<int>(m_lights.size()) - 1);
}

double LightSampler::selection_pdf(int light_index) const {
    if (m_lights.empty()) {
        return 0.0;
    }
    if (m_total_weight <= 0.0) {
        return 1.0 / m_lights.size();
    }
    double previous = light_index == 0 ? 0.0 : m_cdf[light_index - 1];
    return (m_cdf[light_index] - previous) / m_total_weight;
}

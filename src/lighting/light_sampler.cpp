#include "light_sampler.h"

#include <algorithm>
#include <cmath>

namespace {

constexpr double kUniformLightMix = 0.05;

double luminance(const color &value) {
    return 0.2126 * value.x() + 0.7152 * value.y() +
           0.0722 * value.z();
}

} // namespace

LightSampler::LightSampler(const std::vector<shared_ptr<Light>> &lights) {
    for (const auto &light : lights) {
        if (!light) {
            continue;
        }
        (light->is_delta() ? m_delta_lights : m_non_delta_lights)
            .push_back(light);
    }

    if (m_non_delta_lights.empty()) {
        return;
    }

    std::vector<double> weights;
    weights.reserve(m_non_delta_lights.size());
    double total_weight = 0.0;
    for (const auto &light : m_non_delta_lights) {
        double weight = luminance(light->power());
        if (!std::isfinite(weight) || weight < 0.0) {
            weight = 0.0;
        }
        weights.push_back(weight);
        total_weight += weight;
    }

    const double uniform_probability =
        1.0 / static_cast<double>(m_non_delta_lights.size());
    m_selection_probabilities.reserve(m_non_delta_lights.size());
    m_cdf.reserve(m_non_delta_lights.size());
    double accumulated_probability = 0.0;
    for (double weight : weights) {
        const double power_probability =
            total_weight > 0.0 ? weight / total_weight : uniform_probability;
        const double probability =
            (1.0 - kUniformLightMix) * power_probability +
            kUniformLightMix * uniform_probability;
        m_selection_probabilities.push_back(probability);
        accumulated_probability += probability;
        m_cdf.push_back(accumulated_probability);
    }
    m_cdf.back() = 1.0;
}

bool LightSampler::empty() const {
    return m_delta_lights.empty() && m_non_delta_lights.empty();
}

bool LightSampler::has_non_delta_lights() const {
    return !m_non_delta_lights.empty();
}

const std::vector<shared_ptr<Light>> &LightSampler::delta_lights() const {
    return m_delta_lights;
}

SelectedLightSample LightSampler::sample_non_delta(const point3 &p,
                                                   RNG &rng) const {
    SelectedLightSample result;
    if (m_non_delta_lights.empty()) {
        return result;
    }

    const int light_index = choose_light(rng.next());
    const auto &light = m_non_delta_lights[light_index];
    result.sample = light->sample(p, vec2(rng.next(), rng.next()));
    result.selection_pdf = m_selection_probabilities[light_index];
    result.has_bsdf_competitor = light->is_bsdf_hittable();
    return result;
}

double LightSampler::pdf(const point3 &origin, const vec3 &direction) const {
    double total_pdf = 0.0;
    for (std::size_t i = 0; i < m_non_delta_lights.size(); ++i) {
        if (!m_non_delta_lights[i]->is_bsdf_hittable()) {
            continue;
        }
        total_pdf += m_non_delta_lights[i]->pdf(origin, direction) *
                     m_selection_probabilities[i];
    }
    return total_pdf;
}

color LightSampler::environment_radiance(
    const ray &r, bool &found_environment) const {
    color result(0, 0, 0);
    found_environment = false;
    for (const auto &light : m_non_delta_lights) {
        if (light->is_infinite()) {
            result += light->Le(r);
            found_environment = true;
        }
    }
    return result;
}

std::size_t LightSampler::non_delta_light_count() const {
    return m_non_delta_lights.size();
}

double LightSampler::non_delta_selection_pdf(std::size_t light_index) const {
    return light_index < m_selection_probabilities.size()
               ? m_selection_probabilities[light_index]
               : 0.0;
}

int LightSampler::choose_light(double u) const {
    if (m_non_delta_lights.empty()) {
        return 0;
    }
    const double target = clamp(u, 0.0, 0.999999999999);
    auto it = std::lower_bound(m_cdf.begin(), m_cdf.end(), target);
    int index = static_cast<int>(it - m_cdf.begin());
    return std::min(index,
                    static_cast<int>(m_non_delta_lights.size()) - 1);
}

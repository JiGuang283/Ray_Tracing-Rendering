#include "color_pipeline.h"

#include "rtweekend.h"

#include <algorithm>
#include <cmath>

namespace {

color max_zero(const color &c) {
    return color(std::max(0.0, c.x()), std::max(0.0, c.y()),
                 std::max(0.0, c.z()));
}

color apply_reinhard(const color &c) {
    return color(c.x() / (1.0 + c.x()), c.y() / (1.0 + c.y()),
                 c.z() / (1.0 + c.z()));
}

double aces_channel(double x) {
    constexpr double a = 2.51;
    constexpr double b = 0.03;
    constexpr double c = 2.43;
    constexpr double d = 0.59;
    constexpr double e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

color apply_aces(const color &c) {
    return color(aces_channel(c.x()), aces_channel(c.y()),
                 aces_channel(c.z()));
}

color apply_gamma(const color &c, double gamma) {
    if (gamma <= 0.0) {
        return c;
    }
    const double inv_gamma = 1.0 / gamma;
    return color(std::pow(c.x(), inv_gamma), std::pow(c.y(), inv_gamma),
                 std::pow(c.z(), inv_gamma));
}

} // namespace

ColorPipeline::ColorPipeline(const ColorPipelineSettings &settings)
    : m_settings(settings) {
}

void ColorPipeline::set_settings(const ColorPipelineSettings &settings) {
    m_settings = settings;
}

color ColorPipeline::to_display(const color &linear_radiance,
                                std::uint32_t sample_count) const {
    if (sample_count == 0) {
        return color(0, 0, 0);
    }
    const double scale = 1.0 / static_cast<double>(sample_count);
    color mapped = max_zero(scale * linear_radiance);
    mapped *= std::pow(2.0, m_settings.exposure);

    switch (m_settings.tone_mapping) {
    case ToneMappingMode::Linear:
        break;
    case ToneMappingMode::Reinhard:
        mapped = apply_reinhard(mapped);
        break;
    case ToneMappingMode::ACES:
        mapped = apply_aces(mapped);
        break;
    }

    mapped = apply_gamma(max_zero(mapped), m_settings.gamma);
    return color(clamp(mapped.x(), 0.0, 1.0), clamp(mapped.y(), 0.0, 1.0),
                 clamp(mapped.z(), 0.0, 1.0));
}

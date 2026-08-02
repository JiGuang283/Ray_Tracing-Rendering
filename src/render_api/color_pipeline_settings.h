#ifndef COLOR_PIPELINE_SETTINGS_H
#define COLOR_PIPELINE_SETTINGS_H

#include <cstdint>

enum class ToneMappingMode : std::uint32_t {
    Linear,
    Reinhard,
    ACES
};

struct ColorPipelineSettings {
    double exposure = 0.0;
    double gamma = 2.0;
    ToneMappingMode tone_mapping = ToneMappingMode::Linear;
};

#endif

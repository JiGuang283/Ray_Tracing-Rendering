#ifndef COLOR_PIPELINE_H
#define COLOR_PIPELINE_H

#include "rtweekend.h"
#include "vec3.h"

enum class ToneMappingMode {
    Linear,
    Reinhard,
    ACES
};

struct ColorPipelineSettings {
    double exposure = 0.0;
    double gamma = 2.0;
    ToneMappingMode tone_mapping = ToneMappingMode::Linear;
};

class ColorPipeline {
  public:
    ColorPipeline() = default;
    explicit ColorPipeline(const ColorPipelineSettings &settings);

    void set_settings(const ColorPipelineSettings &settings);
    color to_display(const color &linear_radiance, int sample_count) const;

  private:
    ColorPipelineSettings m_settings;
};

#endif

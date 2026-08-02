#ifndef COLOR_PIPELINE_H
#define COLOR_PIPELINE_H

#include "color_pipeline_settings.h"
#include "vec3.h"

#include <cstdint>

class ColorPipeline {
  public:
    ColorPipeline() = default;
    explicit ColorPipeline(const ColorPipelineSettings &settings);

    void set_settings(const ColorPipelineSettings &settings);
    color to_display(const color &linear_radiance,
                     std::uint32_t sample_count) const;

  private:
    ColorPipelineSettings m_settings;
};

#endif

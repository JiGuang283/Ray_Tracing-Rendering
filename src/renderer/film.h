#ifndef FILM_H
#define FILM_H

#include "color_pipeline.h"
#include "render_buffer.h"

#include <vector>

struct FilmPixel {
    color beauty{0, 0, 0};
    color normal{0, 0, 0};
    color albedo{0, 0, 0};
    double depth = 0.0;
    int sample_count = 0;
    int normal_count = 0;
    int albedo_count = 0;
    int depth_count = 0;
};

class Film {
  public:
    explicit Film(RenderBuffer &display_buffer);
    Film(RenderBuffer &display_buffer,
         const ColorPipelineSettings &color_settings);

    int width() const;
    int height() const;

    void add_sample(int x, int y, const color &radiance);
    void add_normal_sample(int x, int y, const vec3 &normal);
    void add_albedo_sample(int x, int y, const color &albedo);
    void add_depth_sample(int x, int y, double depth);
    void finalize_pixel(int x, int y);

    const std::vector<FilmPixel> &pixels() const;

  private:
    int index(int x, int y) const;

    RenderBuffer &m_display_buffer;
    ColorPipeline m_color_pipeline;
    std::vector<FilmPixel> m_pixels;
};

#endif

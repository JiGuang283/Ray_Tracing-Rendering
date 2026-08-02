#ifndef FILM_H
#define FILM_H

#include "beauty_film.h"
#include "color_pipeline.h"
#include "preview_surface.h"

class Film {
  public:
    Film(ImageExtent extent, const ColorPipelineSettings &color_settings,
         PreviewSurface *preview = nullptr);

    int width() const;
    int height() const;

    void add_sample(int x, int y, const color &radiance);
    void finalize_pixel(int x, int y);

    const BeautyFilm &beauty() const noexcept;
    BeautyFilm take_beauty() noexcept;

  private:
    BeautyFilm m_beauty;
    ColorPipeline m_color_pipeline;
    PreviewSurface *m_preview = nullptr;
};

#endif

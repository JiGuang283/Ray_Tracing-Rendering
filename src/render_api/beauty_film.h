#ifndef BEAUTY_FILM_H
#define BEAUTY_FILM_H

#include "color_pipeline_settings.h"
#include "render_buffer.h"
#include "render_types.h"
#include "vec3.h"

#include <cstdint>
#include <string>
#include <vector>

enum class FilmChannel : std::uint32_t {
    Beauty = 1u << 0
};

struct BeautyFilmPixel {
    color radiance_sum{0, 0, 0};
    std::uint32_t sample_count = 0;
};

class BeautyFilm {
  public:
    BeautyFilm() = default;
    explicit BeautyFilm(ImageExtent extent);

    ImageExtent extent() const noexcept;
    int width() const noexcept;
    int height() const noexcept;

    void add_sample(int x, int y, const color &radiance);
    void set_pixel(int x, int y, const color &radiance_sum,
                   std::uint32_t sample_count);
    const BeautyFilmPixel &pixel(int x, int y) const;
    const std::vector<BeautyFilmPixel> &pixels() const noexcept;

    void save_to_pfm(const std::string &filename) const;

  private:
    std::size_t index(int x, int y) const;

    ImageExtent m_extent;
    std::vector<BeautyFilmPixel> m_pixels;
};

RenderBuffer resolve_beauty(const BeautyFilm &film,
                            const ColorPipelineSettings &settings);

#endif

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

    // Checked API. The unchecked variants below are for render backends that
    // have already validated image coordinates and want to keep the hottest
    // per-sample write free of bounds checks.
    void add_sample(int x, int y, const color &radiance);
    void set_pixel(int x, int y, const color &radiance_sum,
                   std::uint32_t sample_count);
    const BeautyFilmPixel &pixel(int x, int y) const;

    void add_sample_unchecked(int x, int y, const color &radiance);
    void set_pixel_unchecked(int x, int y, const color &radiance_sum,
                             std::uint32_t sample_count);
    const BeautyFilmPixel &pixel_unchecked(int x, int y) const;

    const std::vector<BeautyFilmPixel> &pixels() const noexcept;

    void save_to_pfm(const std::string &filename) const;

  private:
    std::size_t index(int x, int y) const;
    std::size_t index_unchecked(int x, int y) const noexcept;

    ImageExtent m_extent;
    std::vector<BeautyFilmPixel> m_pixels;
};

RenderBuffer resolve_beauty(const BeautyFilm &film,
                            const ColorPipelineSettings &settings);

#endif

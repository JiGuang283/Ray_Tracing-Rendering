#include "beauty_film.h"

#include "color_pipeline.h"

#include <stdexcept>

BeautyFilm::BeautyFilm(ImageExtent extent) : m_extent(extent) {
    if (extent.width != 0 || extent.height != 0) {
        make_image_extent(static_cast<int>(extent.width),
                          static_cast<int>(extent.height));
    }
    m_pixels.resize(extent.pixel_count());
}

ImageExtent BeautyFilm::extent() const noexcept {
    return m_extent;
}

int BeautyFilm::width() const noexcept {
    return static_cast<int>(m_extent.width);
}

int BeautyFilm::height() const noexcept {
    return static_cast<int>(m_extent.height);
}

void BeautyFilm::add_sample(int x, int y, const color &radiance) {
    BeautyFilmPixel &film_pixel = m_pixels.at(index(x, y));
    film_pixel.radiance_sum += radiance;
    ++film_pixel.sample_count;
}

void BeautyFilm::set_pixel(int x, int y, const color &radiance_sum,
                           std::uint32_t sample_count) {
    BeautyFilmPixel &film_pixel = m_pixels.at(index(x, y));
    film_pixel.radiance_sum = radiance_sum;
    film_pixel.sample_count = sample_count;
}

const BeautyFilmPixel &BeautyFilm::pixel(int x, int y) const {
    return m_pixels.at(index(x, y));
}

const std::vector<BeautyFilmPixel> &BeautyFilm::pixels() const noexcept {
    return m_pixels;
}

std::size_t BeautyFilm::index(int x, int y) const {
    if (x < 0 || y < 0 || x >= width() || y >= height()) {
        throw std::out_of_range("film pixel is out of range");
    }
    return static_cast<std::size_t>(y) * m_extent.width +
           static_cast<std::size_t>(x);
}

RenderBuffer resolve_beauty(const BeautyFilm &film,
                            const ColorPipelineSettings &settings) {
    RenderBuffer result(film.width(), film.height());
    const ColorPipeline pipeline(settings);
    for (int y = 0; y < film.height(); ++y) {
        for (int x = 0; x < film.width(); ++x) {
            const BeautyFilmPixel &pixel = film.pixel(x, y);
            result.set_pixel(
                x, y,
                pipeline.to_display(pixel.radiance_sum, pixel.sample_count));
        }
    }
    return result;
}

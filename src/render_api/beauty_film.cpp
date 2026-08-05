#include "beauty_film.h"

#include "color_pipeline.h"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

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

void BeautyFilm::save_to_pfm(const std::string &filename) const {
    if (m_extent.width == 0 || m_extent.height == 0) {
        throw std::logic_error("cannot save an empty beauty film");
    }

    std::ofstream output(filename, std::ios::binary);
    if (!output) {
        throw std::runtime_error("failed to open PFM output '" + filename +
                                 "'");
    }

    const std::uint16_t endian_probe = 1;
    const bool little_endian =
        *reinterpret_cast<const unsigned char *>(&endian_probe) == 1;
    output << "PF\n" << m_extent.width << ' ' << m_extent.height << '\n'
           << (little_endian ? "-1.0\n" : "1.0\n");

    std::vector<float> row(static_cast<std::size_t>(m_extent.width) * 3);
    for (std::uint32_t y = 0; y < m_extent.height; ++y) {
        for (std::uint32_t x = 0; x < m_extent.width; ++x) {
            const BeautyFilmPixel &film_pixel =
                m_pixels[static_cast<std::size_t>(y) * m_extent.width + x];
            const double inverse_count =
                film_pixel.sample_count > 0
                    ? 1.0 / static_cast<double>(film_pixel.sample_count)
                    : 0.0;
            for (int channel = 0; channel < 3; ++channel) {
                const double value =
                    film_pixel.radiance_sum[channel] * inverse_count;
                const float packed = static_cast<float>(value);
                if (!std::isfinite(value) || !std::isfinite(packed)) {
                    throw std::runtime_error(
                        "non-finite radiance in PFM output at pixel (" +
                        std::to_string(x) + ", " + std::to_string(y) + ")");
                }
                row[static_cast<std::size_t>(x) * 3 + channel] = packed;
            }
        }
        output.write(reinterpret_cast<const char *>(row.data()),
                     static_cast<std::streamsize>(row.size() * sizeof(float)));
        if (!output) {
            throw std::runtime_error("failed while writing PFM output '" +
                                     filename + "'");
        }
    }
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

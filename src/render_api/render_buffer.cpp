#include "render_buffer.h"

#include <limits>
#include <stdexcept>
#include <vector>

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

RenderBuffer::RenderBuffer(int width, int height)
    : m_width(width), m_height(height) {
    if (width < 0 || height < 0) {
        throw std::invalid_argument("render buffer dimensions cannot be negative");
    }
    const std::size_t pixel_count = static_cast<std::size_t>(width) * height;
    if (height != 0 && pixel_count / static_cast<std::size_t>(height) !=
                           static_cast<std::size_t>(width)) {
        throw std::overflow_error("render buffer dimensions overflow");
    }
    m_pixels.resize(pixel_count);
}

void RenderBuffer::set_pixel(int x, int y, const color &pixel_color) {
    if (x >= 0 && x < m_width && y >= 0 && y < m_height) {
        m_pixels[index(x, y)] = pixel_color;
    }
}

const std::vector<color> &RenderBuffer::get_data() const {
    return m_pixels;
}

int RenderBuffer::width() const {
    return m_width;
}

int RenderBuffer::height() const {
    return m_height;
}

bool RenderBuffer::save_to_png(const std::string &filename) const {
    std::vector<unsigned char> image_data(
        static_cast<std::size_t>(m_width) * m_height * 3);

    for (int j = 0; j < m_height; ++j) {
        for (int i = 0; i < m_width; ++i) {
            const int flipped_j = m_height - 1 - j;
            const int image_index = (j * m_width + i) * 3;
            const auto &pixel = m_pixels[index(i, flipped_j)];
            image_data[image_index + 0] =
                static_cast<unsigned char>(pixel[0] * 255);
            image_data[image_index + 1] =
                static_cast<unsigned char>(pixel[1] * 255);
            image_data[image_index + 2] =
                static_cast<unsigned char>(pixel[2] * 255);
        }
    }

    return stbi_write_png(filename.c_str(), m_width, m_height, 3,
                          image_data.data(), m_width * 3);
}

bool RenderBuffer::save_to_jpg(const std::string &filename,
                               int quality) const {
    std::vector<unsigned char> image_data(
        static_cast<std::size_t>(m_width) * m_height * 3);

    for (int j = 0; j < m_height; ++j) {
        for (int i = 0; i < m_width; ++i) {
            const int flipped_j = m_height - 1 - j;
            const int image_index = (j * m_width + i) * 3;
            const auto &pixel = m_pixels[index(i, flipped_j)];
            image_data[image_index + 0] =
                static_cast<unsigned char>(pixel[0] * 255);
            image_data[image_index + 1] =
                static_cast<unsigned char>(pixel[1] * 255);
            image_data[image_index + 2] =
                static_cast<unsigned char>(pixel[2] * 255);
        }
    }

    return stbi_write_jpg(filename.c_str(), m_width, m_height, 3,
                          image_data.data(), quality);
}

std::size_t RenderBuffer::index(int x, int y) const {
    return static_cast<std::size_t>(y) * m_width + x;
}

#include "render_buffer.h"

#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

bool RenderBuffer::save_to_png(const std::string &filename) const {
    std::vector<unsigned char> image_data(m_width * m_height * 3);

    for (int j = 0; j < m_height; ++j) {
        for (int i = 0; i < m_width; ++i) {
            int flipped_j = m_height - 1 - j;
            int image_index = (j * m_width + i) * 3;
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
    std::vector<unsigned char> image_data(m_width * m_height * 3);

    for (int j = 0; j < m_height; ++j) {
        for (int i = 0; i < m_width; ++i) {
            int flipped_j = m_height - 1 - j;
            int image_index = (j * m_width + i) * 3;
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

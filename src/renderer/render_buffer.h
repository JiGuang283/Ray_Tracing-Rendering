#ifndef RENDER_BUFFER_H
#define RENDER_BUFFER_H

#include "vec3.h"
#include <string>
#include <vector>

class RenderBuffer {
  public:
    RenderBuffer(int width, int height) : m_width(width), m_height(height) {
        m_pixels.resize(static_cast<size_t>(width) * height);
    }

    void set_pixel(int x, int y, const color &pixel_color) {
        if (x >= 0 && x < m_width && y >= 0 && y < m_height) {
            m_pixels[index(x, y)] = pixel_color;
        }
    }

    const std::vector<color> &get_data() const {
        return m_pixels;
    }

    int width() const {
        return m_width;
    }
    int height() const {
        return m_height;
    }

    bool save_to_png(const std::string &filename) const;
    bool save_to_jpg(const std::string &filename, int quality = 90) const;

  private:
    int index(int x, int y) const {
        return y * m_width + x;
    }

    int m_width;
    int m_height;
    std::vector<color> m_pixels;
};

#endif

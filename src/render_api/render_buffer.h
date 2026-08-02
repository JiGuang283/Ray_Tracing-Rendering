#ifndef RENDER_BUFFER_H
#define RENDER_BUFFER_H

#include "vec3.h"

#include <string>
#include <vector>

class RenderBuffer {
  public:
    RenderBuffer() = default;
    RenderBuffer(int width, int height);

    void set_pixel(int x, int y, const color &pixel_color);
    const std::vector<color> &get_data() const;

    int width() const;
    int height() const;

    bool save_to_png(const std::string &filename) const;
    bool save_to_jpg(const std::string &filename, int quality = 90) const;

  private:
    std::size_t index(int x, int y) const;

    int m_width = 0;
    int m_height = 0;
    std::vector<color> m_pixels;
};

#endif

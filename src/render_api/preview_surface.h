#ifndef PREVIEW_SURFACE_H
#define PREVIEW_SURFACE_H

#include "render_buffer.h"
#include "render_types.h"

#include <atomic>
#include <cstdint>
#include <memory>

class PreviewSurface {
  public:
    explicit PreviewSurface(ImageExtent extent);

    ImageExtent extent() const noexcept;
    void publish_pixel(int x, int y, const color &display_color) noexcept;
    void publish(const RenderBuffer &buffer);
    RenderBuffer snapshot() const;

  private:
    std::size_t index(int x, int y) const noexcept;

    ImageExtent m_extent;
    std::unique_ptr<std::atomic<std::uint32_t>[]> m_pixels;
};

#endif

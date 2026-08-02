#include "preview_surface.h"

#include "rtweekend.h"

#include <stdexcept>

namespace {

std::uint32_t pack_color(const color &value) noexcept {
    const auto channel = [](double component) {
        return static_cast<std::uint32_t>(clamp(component, 0.0, 1.0) * 255.0);
    };
    return channel(value.x()) | (channel(value.y()) << 8u) |
           (channel(value.z()) << 16u);
}

color unpack_color(std::uint32_t packed) noexcept {
    constexpr double scale = 1.0 / 255.0;
    return color(static_cast<double>(packed & 0xffu) * scale,
                 static_cast<double>((packed >> 8u) & 0xffu) * scale,
                 static_cast<double>((packed >> 16u) & 0xffu) * scale);
}

} // namespace

PreviewSurface::PreviewSurface(ImageExtent extent) : m_extent(extent) {
    make_image_extent(static_cast<int>(extent.width),
                      static_cast<int>(extent.height));
    m_pixels = std::make_unique<std::atomic<std::uint32_t>[]>(
        extent.pixel_count());
    for (std::size_t index = 0; index < extent.pixel_count(); ++index) {
        m_pixels[index].store(0, std::memory_order_relaxed);
    }
}

ImageExtent PreviewSurface::extent() const noexcept {
    return m_extent;
}

void PreviewSurface::publish_pixel(int x, int y,
                                   const color &display_color) noexcept {
    if (x < 0 || y < 0 || x >= static_cast<int>(m_extent.width) ||
        y >= static_cast<int>(m_extent.height)) {
        return;
    }
    m_pixels[index(x, y)].store(pack_color(display_color),
                                std::memory_order_release);
}

void PreviewSurface::publish(const RenderBuffer &buffer) {
    if (buffer.width() != static_cast<int>(m_extent.width) ||
        buffer.height() != static_cast<int>(m_extent.height)) {
        throw std::invalid_argument("preview and render buffer extents differ");
    }
    const auto &pixels = buffer.get_data();
    for (int y = 0; y < buffer.height(); ++y) {
        for (int x = 0; x < buffer.width(); ++x) {
            publish_pixel(x, y,
                          pixels[static_cast<std::size_t>(y) * m_extent.width +
                                 static_cast<std::size_t>(x)]);
        }
    }
}

RenderBuffer PreviewSurface::snapshot() const {
    RenderBuffer result(static_cast<int>(m_extent.width),
                        static_cast<int>(m_extent.height));
    for (int y = 0; y < static_cast<int>(m_extent.height); ++y) {
        for (int x = 0; x < static_cast<int>(m_extent.width); ++x) {
            const std::uint32_t packed =
                m_pixels[index(x, y)].load(std::memory_order_acquire);
            result.set_pixel(x, y, unpack_color(packed));
        }
    }
    return result;
}

std::size_t PreviewSurface::index(int x, int y) const noexcept {
    return static_cast<std::size_t>(y) * m_extent.width +
           static_cast<std::size_t>(x);
}

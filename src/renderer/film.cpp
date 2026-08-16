#include "film.h"

Film::Film(ImageExtent extent, const ColorPipelineSettings &color_settings,
           PreviewSurface *preview)
    : m_beauty(extent), m_color_pipeline(color_settings), m_preview(preview) {
}

int Film::width() const {
    return m_beauty.width();
}

int Film::height() const {
    return m_beauty.height();
}

void Film::add_sample(int x, int y, const color &radiance) {
    // Film dimensions are validated at construction and the renderer only
    // visits in-bounds pixels, so the unchecked BeautyFilm path is safe.
    m_beauty.add_sample_unchecked(x, y, radiance);
}

void Film::finalize_pixel(int x, int y) {
    if (m_preview == nullptr) {
        return;
    }
    const BeautyFilmPixel &pixel = m_beauty.pixel_unchecked(x, y);
    if (pixel.sample_count != 0) {
        m_preview->publish_pixel(
            x, y,
            m_color_pipeline.to_display(pixel.radiance_sum,
                                        pixel.sample_count));
    }
}

const BeautyFilm &Film::beauty() const noexcept {
    return m_beauty;
}

BeautyFilm Film::take_beauty() noexcept {
    return std::move(m_beauty);
}

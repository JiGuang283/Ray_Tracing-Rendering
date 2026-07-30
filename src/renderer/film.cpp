#include "film.h"

Film::Film(RenderBuffer &display_buffer)
    : Film(display_buffer, ColorPipelineSettings()) {
}

Film::Film(RenderBuffer &display_buffer,
           const ColorPipelineSettings &color_settings)
    : m_display_buffer(display_buffer), m_color_pipeline(color_settings) {
    m_pixels.resize(static_cast<size_t>(width()) * height());
}

int Film::width() const {
    return m_display_buffer.width();
}

int Film::height() const {
    return m_display_buffer.height();
}

void Film::add_sample(int x, int y, const color &radiance) {
    int pixel_index = index(x, y);
    m_pixels[pixel_index].beauty += radiance;
    m_pixels[pixel_index].sample_count += 1;
}

void Film::add_normal_sample(int x, int y, const vec3 &normal) {
    int pixel_index = index(x, y);
    m_pixels[pixel_index].normal += normal;
    m_pixels[pixel_index].normal_count += 1;
}

void Film::add_albedo_sample(int x, int y, const color &albedo) {
    int pixel_index = index(x, y);
    m_pixels[pixel_index].albedo += albedo;
    m_pixels[pixel_index].albedo_count += 1;
}

void Film::add_depth_sample(int x, int y, double depth) {
    int pixel_index = index(x, y);
    m_pixels[pixel_index].depth += depth;
    m_pixels[pixel_index].depth_count += 1;
}

void Film::finalize_pixel(int x, int y) {
    int pixel_index = index(x, y);
    int sample_count = m_pixels[pixel_index].sample_count;
    if (sample_count <= 0) {
        return;
    }
    m_display_buffer.set_pixel(
        x, y,
        m_color_pipeline.to_display(m_pixels[pixel_index].beauty,
                                    sample_count));
}

const std::vector<FilmPixel> &Film::pixels() const {
    return m_pixels;
}

int Film::index(int x, int y) const {
    return y * width() + x;
}

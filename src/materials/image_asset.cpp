#include "image_asset.h"

#include "stb_image.h"

#include <algorithm>
#include <utility>

ImageAsset::ImageAsset(int width, int height, int channels,
                       std::vector<float> pixels, bool hdr)
    : m_width(width), m_height(height), m_channels(channels), m_hdr(hdr),
      m_pixels(std::move(pixels)) {
}

std::shared_ptr<const ImageAsset>
ImageAsset::load(const std::string &path, std::string &error) {
    int width = 0;
    int height = 0;
    int channels = 0;
    const bool hdr = stbi_is_hdr(path.c_str()) != 0;
    std::vector<float> pixels;

    if (hdr) {
        float *data =
            stbi_loadf(path.c_str(), &width, &height, &channels, 0);
        if (!data) {
            error = stbi_failure_reason() ? stbi_failure_reason()
                                          : "unknown decoder error";
            return nullptr;
        }
        pixels.assign(data, data + width * height * channels);
        stbi_image_free(data);
    } else {
        unsigned char *data =
            stbi_load(path.c_str(), &width, &height, &channels, 0);
        if (!data) {
            error = stbi_failure_reason() ? stbi_failure_reason()
                                          : "unknown decoder error";
            return nullptr;
        }
        const std::size_t count =
            static_cast<std::size_t>(width) * height * channels;
        pixels.resize(count);
        for (std::size_t i = 0; i < count; ++i) {
            pixels[i] = static_cast<float>(data[i]) / 255.0f;
        }
        stbi_image_free(data);
    }

    if (width <= 0 || height <= 0 || channels <= 0 || channels > 4) {
        error = "unsupported image dimensions or channel count";
        return nullptr;
    }
    return std::shared_ptr<const ImageAsset>(
        new ImageAsset(width, height, channels, std::move(pixels), hdr));
}

std::shared_ptr<const ImageAsset> ImageAsset::diagnostic() {
    static const std::shared_ptr<const ImageAsset> asset = from_pixels(
        2, 2, 3,
        {
            1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f,
        });
    return asset;
}

std::shared_ptr<const ImageAsset>
ImageAsset::from_pixels(int width, int height, int channels,
                        std::vector<float> pixels, bool hdr) {
    const std::size_t expected =
        static_cast<std::size_t>(width) * height * channels;
    if (width <= 0 || height <= 0 || channels <= 0 || channels > 4 ||
        pixels.size() != expected) {
        return nullptr;
    }
    return std::shared_ptr<const ImageAsset>(
        new ImageAsset(width, height, channels, std::move(pixels), hdr));
}

int ImageAsset::width() const {
    return m_width;
}

int ImageAsset::height() const {
    return m_height;
}

int ImageAsset::channels() const {
    return m_channels;
}

bool ImageAsset::is_hdr() const {
    return m_hdr;
}

float ImageAsset::component(int x, int y, int channel) const {
    x = std::max(0, std::min(x, m_width - 1));
    y = std::max(0, std::min(y, m_height - 1));

    int source_channel = channel;
    if (m_channels == 1) {
        source_channel = 0;
    } else if (m_channels == 2) {
        source_channel = channel == 3 ? 1 : 0;
    } else if (channel >= m_channels) {
        return channel == 3 ? 1.0f : m_pixels[static_cast<std::size_t>(
                                          (y * m_width + x) * m_channels)];
    }

    const std::size_t index =
        static_cast<std::size_t>((y * m_width + x) * m_channels +
                                 source_channel);
    return m_pixels[index];
}

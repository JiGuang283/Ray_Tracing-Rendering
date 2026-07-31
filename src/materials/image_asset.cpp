#include "image_asset.h"

#include "stb_image.h"

#include <algorithm>
#include <climits>
#include <utility>

namespace {

bool valid_dimensions(int width, int height, int channels) {
    return width > 0 && height > 0 && channels > 0 && channels <= 4;
}

std::shared_ptr<const ImageAsset>
decode_memory(const std::uint8_t *data, std::size_t size,
              std::string &error) {
    if (!data || size == 0 || size > static_cast<std::size_t>(INT_MAX)) {
        error = "invalid or oversized image buffer";
        return nullptr;
    }

    int width = 0;
    int height = 0;
    int channels = 0;
    const int byte_count = static_cast<int>(size);
    const bool hdr = stbi_is_hdr_from_memory(data, byte_count) != 0;
    std::vector<float> pixels;
    if (hdr) {
        float *decoded = stbi_loadf_from_memory(
            data, byte_count, &width, &height, &channels, 0);
        if (!decoded) {
            error = stbi_failure_reason() ? stbi_failure_reason()
                                          : "unknown decoder error";
            return nullptr;
        }
        if (!valid_dimensions(width, height, channels)) {
            stbi_image_free(decoded);
            error = "unsupported image dimensions or channel count";
            return nullptr;
        }
        const std::size_t component_count =
            static_cast<std::size_t>(width) *
            static_cast<std::size_t>(height) *
            static_cast<std::size_t>(channels);
        pixels.assign(decoded, decoded + component_count);
        stbi_image_free(decoded);
    } else {
        unsigned char *decoded = stbi_load_from_memory(
            data, byte_count, &width, &height, &channels, 0);
        if (!decoded) {
            error = stbi_failure_reason() ? stbi_failure_reason()
                                          : "unknown decoder error";
            return nullptr;
        }
        if (!valid_dimensions(width, height, channels)) {
            stbi_image_free(decoded);
            error = "unsupported image dimensions or channel count";
            return nullptr;
        }
        const std::size_t component_count =
            static_cast<std::size_t>(width) * height * channels;
        pixels.resize(component_count);
        for (std::size_t index = 0; index < component_count; ++index) {
            pixels[index] = static_cast<float>(decoded[index]) / 255.0f;
        }
        stbi_image_free(decoded);
    }

    return ImageAsset::from_pixels(width, height, channels,
                                   std::move(pixels), hdr);
}

} // namespace

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
        if (!valid_dimensions(width, height, channels)) {
            stbi_image_free(data);
            error = "unsupported image dimensions or channel count";
            return nullptr;
        }
        const std::size_t component_count =
            static_cast<std::size_t>(width) *
            static_cast<std::size_t>(height) *
            static_cast<std::size_t>(channels);
        pixels.assign(data, data + component_count);
        stbi_image_free(data);
    } else {
        unsigned char *data =
            stbi_load(path.c_str(), &width, &height, &channels, 0);
        if (!data) {
            error = stbi_failure_reason() ? stbi_failure_reason()
                                          : "unknown decoder error";
            return nullptr;
        }
        if (!valid_dimensions(width, height, channels)) {
            stbi_image_free(data);
            error = "unsupported image dimensions or channel count";
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

    return std::shared_ptr<const ImageAsset>(
        new ImageAsset(width, height, channels, std::move(pixels), hdr));
}

std::shared_ptr<const ImageAsset>
ImageAsset::load_from_memory(const std::uint8_t *data, std::size_t size,
                             std::string &error) {
    return decode_memory(data, size, error);
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
    if (!valid_dimensions(width, height, channels)) {
        return nullptr;
    }
    const std::size_t expected =
        static_cast<std::size_t>(width) * height * channels;
    if (pixels.size() != expected) {
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
        if (channel == 3) {
            return 1.0f;
        }
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

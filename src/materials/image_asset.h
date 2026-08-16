#ifndef IMAGE_ASSET_H
#define IMAGE_ASSET_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class ImageAsset {
  public:
    static std::shared_ptr<const ImageAsset>
    load(const std::string &path, std::string &error);
    static std::shared_ptr<const ImageAsset>
    load_from_memory(const std::uint8_t *data, std::size_t size,
                     std::string &error);
    static std::shared_ptr<const ImageAsset> diagnostic();
    static std::shared_ptr<const ImageAsset>
    from_pixels(int width, int height, int channels,
                std::vector<float> pixels, bool hdr = false);

    int width() const;
    int height() const;
    int channels() const;
    bool is_hdr() const;
    float component(int x, int y, int channel) const;
    // RGB components decoded to linear light for LDR assets. HDR assets are
    // already linear and share the encoded buffer. Alpha is never decoded.
    float linear_component(int x, int y, int channel) const;
    const std::vector<float> &pixels() const;

  private:
    ImageAsset(int width, int height, int channels,
               std::vector<float> pixels, bool hdr);

    int m_width = 0;
    int m_height = 0;
    int m_channels = 0;
    bool m_hdr = false;
    std::vector<float> m_pixels;
    std::vector<float> m_linear_pixels;
};

#endif

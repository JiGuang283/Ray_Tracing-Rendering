#ifndef IMAGE_ASSET_H
#define IMAGE_ASSET_H

#include <memory>
#include <string>
#include <vector>

class ImageAsset {
  public:
    static std::shared_ptr<const ImageAsset>
    load(const std::string &path, std::string &error);
    static std::shared_ptr<const ImageAsset> diagnostic();
    static std::shared_ptr<const ImageAsset>
    from_pixels(int width, int height, int channels,
                std::vector<float> pixels, bool hdr = false);

    int width() const;
    int height() const;
    int channels() const;
    bool is_hdr() const;
    float component(int x, int y, int channel) const;

  private:
    ImageAsset(int width, int height, int channels,
               std::vector<float> pixels, bool hdr);

    int m_width = 0;
    int m_height = 0;
    int m_channels = 0;
    bool m_hdr = false;
    std::vector<float> m_pixels;
};

#endif

#ifndef RESOURCE_REGISTRY_H
#define RESOURCE_REGISTRY_H

#include "image_asset.h"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>

class ResourceRegistry {
  public:
    std::shared_ptr<const ImageAsset>
    load_image(const std::string &path);

    std::size_t image_count() const;

  private:
    static std::string normalize_path(const std::string &path);

    std::unordered_map<std::string, std::shared_ptr<const ImageAsset>>
        m_images;
};

#endif

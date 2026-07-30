#include "resource_registry.h"

#include <filesystem>
#include <iostream>

std::shared_ptr<const ImageAsset>
ResourceRegistry::load_image(const std::string &path) {
    const std::string normalized = normalize_path(path);
    auto found = m_images.find(normalized);
    if (found != m_images.end()) {
        return found->second;
    }

    std::string error;
    auto image = ImageAsset::load(normalized, error);
    if (!image) {
        std::cerr << "ERROR: Could not load image resource '" << path
                  << "': " << error << ". Using diagnostic texture.\n";
        image = ImageAsset::diagnostic();
    }
    m_images.emplace(normalized, image);
    return image;
}

std::size_t ResourceRegistry::image_count() const {
    return m_images.size();
}

std::string ResourceRegistry::normalize_path(const std::string &path) {
    std::error_code error;
    std::filesystem::path candidate(path);
    if (std::filesystem::exists(candidate, error)) {
        return std::filesystem::weakly_canonical(candidate, error).string();
    }
    error.clear();
    return std::filesystem::absolute(candidate, error)
        .lexically_normal()
        .string();
}

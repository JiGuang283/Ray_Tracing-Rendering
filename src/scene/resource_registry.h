#ifndef RESOURCE_REGISTRY_H
#define RESOURCE_REGISTRY_H

#include "image_asset.h"
#include "mesh_asset.h"
#include "model_asset.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

class ResourceRegistry {
  public:
    std::shared_ptr<const ImageAsset>
    load_image(const std::string &path);
    std::shared_ptr<const ImageAsset>
    load_image_from_memory(const std::string &resource_key,
                           const std::uint8_t *data, std::size_t size);
    std::shared_ptr<const ModelAsset>
    load_model(const std::string &path, const ModelImportOptions &options,
               std::string &error);
    std::shared_ptr<const MeshAsset>
    load_obj(const std::string &path, bool build_bvh,
             bool use_vertex_normals, std::string &error);

    std::size_t image_count() const;
    std::size_t model_count() const;
    std::size_t mesh_count() const;

  private:
    static std::string normalize_path(const std::string &path);

    std::unordered_map<std::string, std::shared_ptr<const ImageAsset>>
        m_images;
    std::unordered_map<std::string, std::shared_ptr<const ModelAsset>>
        m_models;
    std::unordered_map<std::string, std::shared_ptr<const MeshAsset>> m_meshes;
};

#endif

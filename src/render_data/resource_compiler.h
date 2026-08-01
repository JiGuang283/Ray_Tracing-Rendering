#ifndef RESOURCE_COMPILER_H
#define RESOURCE_COMPILER_H

#include "compiled_scene.h"
#include "material.h"

#include <unordered_map>
#include <unordered_set>

class PackedResourceCompiler {
  public:
    explicit PackedResourceCompiler(CompiledScene &scene);

    TextureId compile_texture(const TextureHandle &texture);
    MaterialId compile_material(const MaterialHandle &material);

  private:
    ImageId compile_image(const std::shared_ptr<const ImageAsset> &image);
    PerlinId compile_perlin(const perlin &noise);

    CompiledScene &m_scene;
    std::unordered_map<const Texture *, TextureId> m_textures;
    std::unordered_set<const Texture *> m_textures_in_progress;
    std::unordered_map<const MaterialInstance *, MaterialId> m_materials;
    std::unordered_map<const ImageAsset *, ImageId> m_images;
    std::unordered_map<const perlin *, PerlinId> m_perlin;
};

#endif

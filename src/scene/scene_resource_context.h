#ifndef SCENE_RESOURCE_CONTEXT_H
#define SCENE_RESOURCE_CONTEXT_H

#include "material.h"
#include "resource_registry.h"
#include "scene_ir.h"
#include "texture.h"

#include <map>
#include <set>
#include <string>

namespace scene_loader_internal {

using MaterialMap = std::map<std::string, MaterialHandle>;
using TextureCacheKey = std::pair<TextureIRId, TextureSemantic>;

struct SceneResourceContext {
    std::string source_path;
    const SceneIR *scene_ir = nullptr;
    bool strict_assets = false;
    std::map<TextureCacheKey, TextureHandle> textures;
    std::set<TextureCacheKey> textures_in_progress;
    MaterialMap materials;
    ResourceRegistry resources;
};

TextureHandle build_texture(TextureIRId id, TextureSemantic semantic,
                            SceneResourceContext &context);
MaterialHandle build_material(const MaterialIR &material,
                              SceneResourceContext &context);
MaterialHandle lookup_material(SceneResourceContext &context,
                               const std::string &name,
                               const std::string &context_name);

} // namespace scene_loader_internal

#endif

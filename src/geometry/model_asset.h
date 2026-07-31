#ifndef MODEL_ASSET_H
#define MODEL_ASSET_H

#include "material.h"
#include "mesh_asset.h"
#include "transform.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

struct ModelImportOptions {
    bool build_bvh = true;
    bool generate_normals = true;
    bool generate_tangents = true;
};

struct ModelMesh {
    std::string name;
    std::shared_ptr<const MeshAsset> geometry;
};

struct ModelNode {
    std::string name;
    Transform local_transform;
    int mesh_index = -1;
    std::vector<std::size_t> children;
};

struct ModelScene {
    std::string name;
    std::vector<std::size_t> roots;
};

class ModelAsset {
  public:
    ModelAsset(std::vector<ModelMesh> meshes,
               std::vector<ModelNode> nodes,
               std::vector<ModelScene> scenes, int default_scene,
               std::vector<std::string> material_names,
               std::vector<MaterialHandle> materials);

    const std::vector<ModelMesh> &meshes() const;
    const std::vector<ModelNode> &nodes() const;
    const std::vector<ModelScene> &scenes() const;
    const std::vector<std::string> &material_names() const;
    const std::vector<MaterialHandle> &materials() const;
    int default_scene_index() const;
    int resolve_scene_index(int requested_scene) const;

  private:
    std::vector<ModelMesh> m_meshes;
    std::vector<ModelNode> m_nodes;
    std::vector<ModelScene> m_scenes;
    int m_default_scene = 0;
    std::vector<std::string> m_material_names;
    std::vector<MaterialHandle> m_materials;
};

#endif

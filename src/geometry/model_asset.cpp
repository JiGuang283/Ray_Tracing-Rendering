#include "model_asset.h"

#include <stdexcept>
#include <utility>

ModelAsset::ModelAsset(std::vector<ModelMesh> meshes,
                       std::vector<ModelNode> nodes,
                       std::vector<ModelScene> scenes, int default_scene,
                       std::vector<std::string> material_names,
                       std::vector<MaterialHandle> materials)
    : m_meshes(std::move(meshes)), m_nodes(std::move(nodes)),
      m_scenes(std::move(scenes)), m_default_scene(default_scene),
      m_material_names(std::move(material_names)),
      m_materials(std::move(materials)) {
    if (m_scenes.empty()) {
        throw std::invalid_argument("ModelAsset requires at least one scene.");
    }
    if (m_default_scene < 0 ||
        m_default_scene >= static_cast<int>(m_scenes.size())) {
        throw std::invalid_argument("ModelAsset has an invalid default scene.");
    }
    if (m_material_names.size() != m_materials.size()) {
        throw std::invalid_argument(
            "ModelAsset material names and bindings must have equal size.");
    }
    for (const ModelMesh &mesh : m_meshes) {
        if (!mesh.geometry) {
            throw std::invalid_argument(
                "ModelAsset contains a null mesh resource.");
        }
        if (mesh.geometry->material_slot_count() > m_materials.size()) {
            throw std::invalid_argument(
                "ModelAsset mesh references an invalid material slot.");
        }
    }
    for (const ModelNode &node : m_nodes) {
        if (node.mesh_index < -1 ||
            node.mesh_index >= static_cast<int>(m_meshes.size())) {
            throw std::invalid_argument(
                "ModelAsset node references an invalid mesh.");
        }
        for (std::size_t child : node.children) {
            if (child >= m_nodes.size()) {
                throw std::invalid_argument(
                    "ModelAsset node references an invalid child.");
            }
        }
    }
    for (const ModelScene &scene : m_scenes) {
        for (std::size_t root : scene.roots) {
            if (root >= m_nodes.size()) {
                throw std::invalid_argument(
                    "ModelAsset scene references an invalid root node.");
            }
        }
    }
}

const std::vector<ModelMesh> &ModelAsset::meshes() const {
    return m_meshes;
}

const std::vector<ModelNode> &ModelAsset::nodes() const {
    return m_nodes;
}

const std::vector<ModelScene> &ModelAsset::scenes() const {
    return m_scenes;
}

const std::vector<std::string> &ModelAsset::material_names() const {
    return m_material_names;
}

const std::vector<MaterialHandle> &ModelAsset::materials() const {
    return m_materials;
}

int ModelAsset::default_scene_index() const {
    return m_default_scene;
}

int ModelAsset::resolve_scene_index(int requested_scene) const {
    const int result = requested_scene < 0 ? m_default_scene : requested_scene;
    if (result < 0 || result >= static_cast<int>(m_scenes.size())) {
        throw std::out_of_range("Requested model scene index is out of range.");
    }
    return result;
}

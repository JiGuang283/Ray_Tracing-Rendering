#include "scenes.h"

#include "json.hpp"
#include "scene_loader.h"

#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

using json = nlohmann::json;

namespace {

constexpr const char *kSceneCatalogPath = "assets/scenes/catalog.json";
constexpr int kFallbackDefaultSceneId = 10;

struct SceneCatalog {
    int default_scene_id = kFallbackDefaultSceneId;
    std::map<int, std::string> scene_paths;
};

SceneCatalog load_scene_catalog() {
    std::ifstream input(kSceneCatalogPath);
    if (!input) {
        throw std::runtime_error(
            "Scene catalog error: failed to open assets/scenes/catalog.json.");
    }

    json root;
    input >> root;

    SceneCatalog catalog;
    if (root.contains("default_scene_id")) {
        catalog.default_scene_id = root["default_scene_id"].get<int>();
    }

    auto scenes = root.find("scenes");
    if (scenes == root.end() || !scenes->is_array()) {
        throw std::runtime_error(
            "Scene catalog error: 'scenes' must be an array.");
    }

    for (const auto &scene : *scenes) {
        if (!scene.contains("id") || !scene.contains("path")) {
            throw std::runtime_error(
                "Scene catalog error: every scene needs 'id' and 'path'.");
        }

        int scene_id = scene["id"].get<int>();
        std::string path = scene["path"].get<std::string>();
        auto inserted = catalog.scene_paths.emplace(scene_id, path);
        if (!inserted.second) {
            throw std::runtime_error(
                "Scene catalog error: duplicate scene id " +
                std::to_string(scene_id) + ".");
        }
    }

    if (catalog.scene_paths.empty()) {
        throw std::runtime_error("Scene catalog error: no scenes registered.");
    }
    if (catalog.scene_paths.find(catalog.default_scene_id) ==
        catalog.scene_paths.end()) {
        throw std::runtime_error(
            "Scene catalog error: default_scene_id is not present in scenes.");
    }

    return catalog;
}

const SceneCatalog &scene_catalog() {
    static const SceneCatalog catalog = load_scene_catalog();
    return catalog;
}

} // namespace

SceneConfig select_scene(int scene_id) {
    const SceneCatalog &catalog = scene_catalog();
    auto found = catalog.scene_paths.find(scene_id);
    if (found == catalog.scene_paths.end()) {
        found = catalog.scene_paths.find(catalog.default_scene_id);
    }
    return load_scene_file(found->second);
}

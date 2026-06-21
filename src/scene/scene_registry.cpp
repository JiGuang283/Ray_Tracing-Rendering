#include "scene_registry.h"

#include <stdexcept>
#include <utility>

SceneRegistry &SceneRegistry::instance() {
    static SceneRegistry registry;
    return registry;
}

void SceneRegistry::register_scene(int scene_id, SceneFactory factory) {
    factories[scene_id] = std::move(factory);
}

void SceneRegistry::set_default(SceneFactory factory) {
    default_factory = std::move(factory);
}

SceneConfig SceneRegistry::create(int scene_id) const {
    auto found = factories.find(scene_id);
    if (found != factories.end()) {
        return found->second();
    }
    if (default_factory) {
        return default_factory();
    }
    throw std::runtime_error("No scene registered for requested scene id.");
}

static void ensure_builtin_scenes_registered() {
    static bool registered = false;
    if (!registered) {
        register_builtin_scenes(SceneRegistry::instance());
        registered = true;
    }
}

SceneConfig select_scene(int scene_id) {
    ensure_builtin_scenes_registered();
    return SceneRegistry::instance().create(scene_id);
}

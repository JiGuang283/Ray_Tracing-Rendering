#ifndef SCENE_REGISTRY_H
#define SCENE_REGISTRY_H

#include "scene_config.h"
#include <functional>
#include <map>

using SceneFactory = std::function<SceneConfig()>;

class SceneRegistry {
  public:
    static SceneRegistry &instance();

    void register_scene(int scene_id, SceneFactory factory);
    void set_default(SceneFactory factory);
    SceneConfig create(int scene_id) const;

  private:
    std::map<int, SceneFactory> factories;
    SceneFactory default_factory;
};

void register_builtin_scenes(SceneRegistry &registry);

#endif

#include "scene_loader_internal.h"

#include "directional_light.h"
#include "environmental_light.h"
#include "point_light.h"
#include "quad_light.h"
#include "spot_light.h"

#include <stdexcept>

namespace scene_loader_internal {

void add_light(const json &light_json, SceneConfig &config) {
    std::string type = read_string(light_json, "type", "light");
    if (type == "point") {
        config.scene.lights.push_back(make_shared<PointLight>(
            read_vec3(light_json, "position", "point light"),
            read_vec3(light_json, "intensity", "point light")));
        return;
    }
    if (type == "directional") {
        config.scene.lights.push_back(make_shared<DirectionalLight>(
            read_vec3(light_json, "direction", "directional light"),
            read_vec3(light_json, "color", "directional light")));
        return;
    }
    if (type == "spot") {
        config.scene.lights.push_back(make_shared<SpotLight>(
            read_vec3(light_json, "position", "spot light"),
            read_vec3(light_json, "direction", "spot light"),
            read_double_or(light_json, "cutoff", 20.0),
            read_vec3(light_json, "intensity", "spot light")));
        return;
    }
    if (type == "quad") {
        config.scene.lights.push_back(make_shared<QuadLight>(
            read_vec3(light_json, "Q", "quad light"),
            read_vec3(light_json, "u", "quad light"),
            read_vec3(light_json, "v", "quad light"),
            read_vec3(light_json, "intensity", "quad light")));
        return;
    }
    if (type == "environment") {
        std::string path = read_string(light_json, "path", "environment light");
        config.scene.lights.push_back(
            make_shared<EnvironmentLight>(path.c_str()));
        return;
    }

    throw std::runtime_error("Scene file error: unknown light type '" + type +
                             "'.");
}


} // namespace scene_loader_internal

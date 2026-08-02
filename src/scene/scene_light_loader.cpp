#include "scene_loader_internal.h"

#include "asset_path.h"
#include "directional_light.h"
#include "environmental_light.h"
#include "point_light.h"
#include "quad_light.h"
#include "spot_light.h"

#include <stdexcept>
#include <type_traits>

namespace scene_loader_internal {

void add_light(const LightIR &light, SceneBuildContext &context,
               SceneConfig &config) {
    std::visit(
        [&](const auto &typed) {
            using T = std::decay_t<decltype(typed)>;
            if constexpr (std::is_same_v<T, PointLightIR>) {
                config.scene.lights.push_back(make_shared<PointLight>(
                    typed.position, typed.intensity));
            } else if constexpr (std::is_same_v<T, DirectionalLightIR>) {
                config.scene.lights.push_back(make_shared<DirectionalLight>(
                    typed.direction, typed.radiance));
            } else if constexpr (std::is_same_v<T, SpotLightIR>) {
                config.scene.lights.push_back(make_shared<SpotLight>(
                    typed.position, typed.direction, typed.cutoff,
                    typed.intensity));
            } else if constexpr (std::is_same_v<T, QuadLightIR>) {
                config.scene.lights.push_back(make_shared<QuadLight>(
                    typed.origin, typed.u, typed.v, typed.intensity));
            } else if constexpr (std::is_same_v<T, EnvironmentLightIR>) {
                const std::string path =
                    resolve_asset_path(context.source_path, typed.path);
                config.scene.lights.push_back(make_shared<EnvironmentLight>(
                    context.resources.load_image(path)));
            } else {
                throw std::runtime_error(
                    "Scene build error: unsupported LightIR node in " +
                    light.context + ".");
            }
        },
        light.data);
}

} // namespace scene_loader_internal

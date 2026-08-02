#ifndef SCENE_LOADER_INTERNAL_H
#define SCENE_LOADER_INTERNAL_H

#include "hittable.h"
#include "hittable_list.h"
#include "scene_config.h"
#include "scene_json.h"
#include "scene_resource_context.h"

#include <vector>

namespace scene_loader_internal {

using SceneBuildContext = SceneResourceContext;
using scene_json::json;
using scene_json::read_bool_or;
using scene_json::read_double_or;
using scene_json::read_int_or;
using scene_json::read_string;
using scene_json::read_vec2_value;
using scene_json::read_vec3;
using scene_json::read_vec3_or;
using scene_json::read_vec3_value;
using scene_json::require_key;

struct BuiltObject {
    shared_ptr<hittable> object;
    std::vector<shared_ptr<Light>> emitters;
};

shared_ptr<hittable> build_object(ObjectIRId id,
                                  SceneBuildContext &context);
BuiltObject build_object_with_emitters(ObjectIRId id,
                                       SceneBuildContext &context,
                                       bool auto_emitters);
void add_object(ObjectIRId id, SceneBuildContext &context,
                hittable_list &world,
                std::vector<shared_ptr<Light>> &emitters,
                bool auto_emitters);
void add_light(const LightIR &light, SceneBuildContext &context,
               SceneConfig &config);

} // namespace scene_loader_internal

#endif

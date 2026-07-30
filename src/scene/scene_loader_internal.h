#ifndef SCENE_LOADER_INTERNAL_H
#define SCENE_LOADER_INTERNAL_H

#include "hittable.h"
#include "hittable_list.h"
#include "json.hpp"
#include "material.h"
#include "scene_config.h"
#include "texture.h"
#include "vec3.h"

#include <map>
#include <string>

namespace scene_loader_internal {

using json = nlohmann::json;
using MaterialMap = std::map<std::string, shared_ptr<material>>;
using TextureMap = std::map<std::string, shared_ptr<texture>>;

struct SceneBuildContext {
    std::string source_path;
    std::map<std::string, json> texture_specs;
    std::map<std::string, json> material_specs;
    TextureMap textures;
    MaterialMap materials;
};

struct BuiltObject {
    shared_ptr<hittable> object;
    std::vector<shared_ptr<Light>> emitters;
};

const json &require_key(const json &object, const std::string &key,
                        const std::string &context);
vec3 read_vec3_value(const json &value, const std::string &context);
vec2 read_vec2_value(const json &value, const std::string &context);
vec3 read_vec3(const json &object, const std::string &key,
               const std::string &context);
vec3 read_vec3_or(const json &object, const std::string &key,
                  const vec3 &fallback, const std::string &context);
double read_double_or(const json &object, const std::string &key,
                      double fallback);
int read_int_or(const json &object, const std::string &key, int fallback);
bool read_bool_or(const json &object, const std::string &key, bool fallback);
std::string read_string(const json &object, const std::string &key,
                        const std::string &context);
vec2 read_optional_uv(const json &object, const std::string &key,
                      bool &present, const std::string &context);
std::string resolve_asset_path(const SceneBuildContext &context,
                               const std::string &path);

shared_ptr<texture> build_texture_value(const json &texture_json,
                                        SceneBuildContext &context,
                                        const std::string &context_name);
shared_ptr<material> build_material(const json &material_json,
                                    SceneBuildContext &context,
                                    const std::string &name);
shared_ptr<texture> lookup_texture(SceneBuildContext &context,
                                   const std::string &name,
                                   const std::string &context_name);
shared_ptr<material> lookup_material(SceneBuildContext &context,
                                     const json &object,
                                     const std::string &context_name);

shared_ptr<hittable> build_object(const json &object,
                                  SceneBuildContext &context);
BuiltObject build_object_with_emitters(const json &object,
                                       SceneBuildContext &context,
                                       bool auto_emitters);
void add_object(const json &object, SceneBuildContext &context,
                hittable_list &world,
                std::vector<shared_ptr<Light>> &emitters,
                bool auto_emitters);
void add_light(const json &light_json, SceneBuildContext &context,
               SceneConfig &config);

} // namespace scene_loader_internal

#endif

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

shared_ptr<texture> build_texture_value(const json &texture_json,
                                        const TextureMap &textures,
                                        const std::string &context);
shared_ptr<material> build_material(const json &material_json,
                                    const TextureMap &textures,
                                    const std::string &name);
shared_ptr<material> lookup_material(const MaterialMap &materials,
                                     const json &object,
                                     const std::string &context);

shared_ptr<hittable> build_object(const json &object,
                                  const MaterialMap &materials,
                                  const TextureMap &textures);
void add_object(const json &object, const MaterialMap &materials,
                const TextureMap &textures, hittable_list &world);
void add_light(const json &light_json, SceneConfig &config);

} // namespace scene_loader_internal

#endif

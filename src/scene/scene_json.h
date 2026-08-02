#ifndef SCENE_JSON_H
#define SCENE_JSON_H

#include "json.hpp"
#include "vec3.h"

#include <string>

namespace scene_json {

using json = nlohmann::json;

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

} // namespace scene_json

#endif

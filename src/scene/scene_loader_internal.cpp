#include "scene_loader_internal.h"

#include <fstream>
#include <stdexcept>

namespace scene_loader_internal {
namespace {

bool file_exists(const std::string &path) {
    std::ifstream input(path);
    return input.good();
}

std::string parent_path(const std::string &path) {
    size_t slash = path.find_last_of("/\\");
    if (slash == std::string::npos) {
        return "";
    }
    return path.substr(0, slash);
}

bool is_absolute_path(const std::string &path) {
    return !path.empty() && (path[0] == '/' || path[0] == '\\' ||
                             (path.size() > 1 && path[1] == ':'));
}

} // namespace

const json &require_key(const json &object, const std::string &key,
                        const std::string &context) {
    auto found = object.find(key);
    if (found == object.end()) {
        throw std::runtime_error("Scene file error: missing '" + key +
                                 "' in " + context + ".");
    }
    return *found;
}

vec3 read_vec3_value(const json &value, const std::string &context) {
    if (!value.is_array() || value.size() != 3) {
        throw std::runtime_error("Scene file error: expected 3-number array for " +
                                 context + ".");
    }
    return vec3(value[0].get<double>(), value[1].get<double>(),
                value[2].get<double>());
}

vec2 read_vec2_value(const json &value, const std::string &context) {
    if (!value.is_array() || value.size() != 2) {
        throw std::runtime_error("Scene file error: expected 2-number array for " +
                                 context + ".");
    }
    return vec2(value[0].get<double>(), value[1].get<double>());
}

vec3 read_vec3(const json &object, const std::string &key,
               const std::string &context) {
    return read_vec3_value(require_key(object, key, context), context + "." +
                                                               key);
}

vec3 read_vec3_or(const json &object, const std::string &key,
                  const vec3 &fallback, const std::string &context) {
    auto found = object.find(key);
    if (found == object.end()) {
        return fallback;
    }
    return read_vec3_value(*found, context + "." + key);
}

double read_double_or(const json &object, const std::string &key,
                      double fallback) {
    auto found = object.find(key);
    return found == object.end() ? fallback : found->get<double>();
}

int read_int_or(const json &object, const std::string &key, int fallback) {
    auto found = object.find(key);
    return found == object.end() ? fallback : found->get<int>();
}

bool read_bool_or(const json &object, const std::string &key, bool fallback) {
    auto found = object.find(key);
    return found == object.end() ? fallback : found->get<bool>();
}

std::string read_string(const json &object, const std::string &key,
                        const std::string &context) {
    return require_key(object, key, context).get<std::string>();
}

std::string resolve_asset_path(const SceneBuildContext &context,
                               const std::string &path) {
    if (path.empty() || is_absolute_path(path) || file_exists(path)) {
        return path;
    }

    std::string base = parent_path(context.source_path);
    if (base.empty()) {
        return path;
    }

    std::string candidate = base + "/" + path;
    return file_exists(candidate) ? candidate : path;
}

} // namespace scene_loader_internal

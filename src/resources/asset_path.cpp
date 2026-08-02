#include "asset_path.h"

#include <fstream>

namespace {

bool file_exists(const std::string &path) {
    std::ifstream input(path);
    return input.good();
}

std::string parent_path(const std::string &path) {
    const std::size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? std::string() : path.substr(0, slash);
}

bool is_absolute_path(const std::string &path) {
    return !path.empty() &&
           (path[0] == '/' || path[0] == '\\' ||
            (path.size() > 1 && path[1] == ':'));
}

} // namespace

std::string resolve_asset_path(const std::string &source_path,
                               const std::string &asset_path) {
    if (asset_path.empty() || is_absolute_path(asset_path) ||
        file_exists(asset_path)) {
        return asset_path;
    }

    const std::string base = parent_path(source_path);
    if (base.empty()) {
        return asset_path;
    }

    const std::string candidate = base + "/" + asset_path;
    return file_exists(candidate) ? candidate : asset_path;
}

#include "external/json.hpp"
#include "render_data/scene_compiler.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct Options {
    std::filesystem::path catalog = "assets/scenes/catalog.json";
};

Options parse_options(int argc, char **argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--catalog") {
            if (++index >= argc) {
                throw std::runtime_error("--catalog requires a path");
            }
            options.catalog = argv[index];
        } else if (argument == "--help" || argument == "-h") {
            std::cout << "Usage: scene_data_check [--catalog PATH]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + argument);
        }
    }
    return options;
}

nlohmann::json load_json(const std::filesystem::path &path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open catalog: " + path.string());
    }
    nlohmann::json json;
    input >> json;
    return json;
}

std::filesystem::path resolve_scene_path(
    const std::filesystem::path &catalog,
    const std::filesystem::path &scene_path) {
    if (scene_path.is_absolute() || std::filesystem::exists(scene_path)) {
        return scene_path;
    }
    const std::filesystem::path relative = catalog.parent_path() / scene_path;
    if (std::filesystem::exists(relative)) {
        return relative;
    }
    return scene_path;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Options options = parse_options(argc, argv);
        const nlohmann::json catalog = load_json(options.catalog);
        if (!catalog.contains("scenes") || !catalog["scenes"].is_array()) {
            throw std::runtime_error("catalog must contain a scenes array");
        }

        std::size_t passed = 0;
        std::size_t failed = 0;
        std::uint64_t total_bytes = 0;
        for (const nlohmann::json &entry : catalog["scenes"]) {
            const int id = entry.at("id").get<int>();
            const std::string name = entry.at("name").get<std::string>();
            const std::filesystem::path path = resolve_scene_path(
                options.catalog,
                entry.at("path").get<std::string>());
            try {
                const CompiledScene scene = load_compiled_scene(path.string());
                const ValidationReport validation =
                    validate_compiled_scene(scene);
                if (!validation.ok()) {
                    throw std::runtime_error(validation.errors.front());
                }
                const CompiledSceneStats stats = compiled_scene_stats(scene);
                total_bytes += stats.bytes;
                ++passed;
                std::cout << "SCENE_DATA id=" << id << " name=" << name
                          << " nodes=" << stats.bvh_nodes
                          << " triangles=" << stats.triangles
                          << " instances=" << stats.instances
                          << " textures=" << stats.textures
                          << " lights=" << stats.lights
                          << " bytes=" << stats.bytes << '\n';
            } catch (const std::exception &error) {
                ++failed;
                std::cerr << "SCENE_DATA_ERROR id=" << id
                          << " name=" << name << " error=" << error.what()
                          << '\n';
            }
        }

        std::cout << "SCENE_DATA_SUMMARY passed=" << passed
                  << " failed=" << failed << " total_bytes=" << total_bytes
                  << " total_mib=" << std::fixed << std::setprecision(2)
                  << (static_cast<double>(total_bytes) / (1024.0 * 1024.0))
                  << '\n';
        return failed == 0 ? 0 : 1;
    } catch (const std::exception &error) {
        std::cerr << "scene_data_check: " << error.what() << '\n';
        return 1;
    }
}

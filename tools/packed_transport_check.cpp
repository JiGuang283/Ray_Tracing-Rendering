#include "json.hpp"
#include "packed_transport.h"
#include "scene_compiler.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

struct Options {
    std::filesystem::path catalog = "assets/scenes/catalog.json";
    std::set<int> scene_ids{1, 7, 8, 23, 59, 62, 64};
    std::uint32_t width = 16;
    std::uint32_t spp = 2;
    std::uint32_t max_depth = 8;
    std::uint32_t seed = 123;
    PackedIntegratorType integrator = PackedIntegratorType::MISPath;
    bool all = false;
};

std::set<int> parse_ids(const std::string &value) {
    std::set<int> result;
    std::istringstream stream(value);
    std::string part;
    while (std::getline(stream, part, ',')) {
        result.insert(std::stoi(part));
    }
    if (result.empty()) {
        throw std::runtime_error("--ids requires at least one scene ID");
    }
    return result;
}

std::uint32_t positive(const std::string &value,
                       const std::string &option) {
    const unsigned long parsed = std::stoul(value);
    if (parsed == 0 || parsed > 0xfffffffful) {
        throw std::runtime_error(option + " must be a positive uint32");
    }
    return static_cast<std::uint32_t>(parsed);
}

Options parse_options(int argc, char **argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto require_value = [&]() -> std::string {
            if (++index >= argc) {
                throw std::runtime_error(argument + " requires a value");
            }
            return argv[index];
        };
        if (argument == "--catalog") {
            options.catalog = require_value();
        } else if (argument == "--ids") {
            options.scene_ids = parse_ids(require_value());
            options.all = false;
        } else if (argument == "--all") {
            options.all = true;
        } else if (argument == "--width") {
            options.width = positive(require_value(), argument);
        } else if (argument == "--spp") {
            options.spp = positive(require_value(), argument);
        } else if (argument == "--max-depth") {
            options.max_depth = positive(require_value(), argument);
        } else if (argument == "--seed") {
            options.seed = positive(require_value(), argument);
        } else if (argument == "--integrator") {
            const unsigned long parsed = std::stoul(require_value());
            const std::uint32_t value = static_cast<std::uint32_t>(parsed);
            if (parsed > 4) {
                throw std::runtime_error("--integrator must be 0 through 4");
            }
            options.integrator = static_cast<PackedIntegratorType>(value);
        } else if (argument == "--help" || argument == "-h") {
            std::cout << "Usage: packed_transport_check [--catalog PATH] "
                         "[--all] [--ids 1,7,...] [--width N] [--spp N] "
                         "[--max-depth N] [--integrator 0..4] [--seed N]\n";
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
    nlohmann::json result;
    input >> result;
    return result;
}

std::filesystem::path resolve_scene_path(
    const std::filesystem::path &catalog,
    const std::filesystem::path &scene_path) {
    if (scene_path.is_absolute() || std::filesystem::exists(scene_path)) {
        return scene_path;
    }
    const std::filesystem::path relative = catalog.parent_path() / scene_path;
    return std::filesystem::exists(relative) ? relative : scene_path;
}

float length(Float3 value) {
    return std::sqrt(value.x * value.x + value.y * value.y +
                     value.z * value.z);
}

std::uint64_t hash_float(std::uint64_t hash, float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    hash ^= bits;
    return hash * 1099511628211ull;
}

struct CheckResult {
    std::uint64_t samples = 0;
    std::uint64_t shadow_rays = 0;
    std::uint64_t path_depth = 0;
    std::uint64_t errors = 0;
    std::uint64_t checksum = 1469598103934665603ull;
    double average_luminance = 0.0;
};

CheckResult check_scene(const CompiledScene &scene, const Options &options) {
    CheckResult result;
    const float vertical = length(scene.camera.vertical);
    const float aspect = vertical > 0.0f
                             ? length(scene.camera.horizontal) / vertical
                             : 1.0f;
    const std::uint32_t height = std::max(
        1u, static_cast<std::uint32_t>(options.width /
                                      std::max(aspect, 1e-6f)));
    PackedTransportSettings settings;
    settings.integrator = options.integrator;
    settings.max_depth = options.max_depth;
    settings.rr_start_depth = 3;
    const CompiledSceneView view = make_scene_view(scene);
    double luminance_sum = 0.0;
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < options.width; ++x) {
            const std::uint32_t pixel = y * options.width + x;
            for (std::uint32_t sample = 0; sample < options.spp; ++sample) {
                const std::uint32_t sequence =
                    pixel * options.spp + sample + 1;
                RNG rng(mix_seed(options.seed, sequence));
                const PackedRay ray = generate_packed_camera_ray(
                    scene.camera, x, y, options.width, height, rng);
                const PackedTransportResult path =
                    trace_packed_path(view, ray, settings, rng);
                ++result.samples;
                result.shadow_rays += path.shadow_rays;
                result.path_depth += path.depth;
                if (path.status != PackedTransportStatus::Success ||
                    !std::isfinite(path.radiance.x) ||
                    !std::isfinite(path.radiance.y) ||
                    !std::isfinite(path.radiance.z)) {
                    ++result.errors;
                    continue;
                }
                result.checksum = hash_float(result.checksum, path.radiance.x);
                result.checksum = hash_float(result.checksum, path.radiance.y);
                result.checksum = hash_float(result.checksum, path.radiance.z);
                luminance_sum += 0.2126 * path.radiance.x +
                                 0.7152 * path.radiance.y +
                                 0.0722 * path.radiance.z;
            }
        }
    }
    result.average_luminance =
        result.samples != 0 ? luminance_sum / result.samples : 0.0;
    return result;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Options options = parse_options(argc, argv);
        const nlohmann::json catalog = load_json(options.catalog);
        std::size_t selected = 0;
        std::size_t passed = 0;
        std::size_t failed = 0;
        for (const nlohmann::json &entry : catalog.at("scenes")) {
            const int id = entry.at("id").get<int>();
            if (!options.all && options.scene_ids.count(id) == 0) {
                continue;
            }
            ++selected;
            const std::filesystem::path path = resolve_scene_path(
                options.catalog, entry.at("path").get<std::string>());
            const CompiledScene scene = load_compiled_scene(path.string());
            const CheckResult result = check_scene(scene, options);
            const bool ok = result.errors == 0;
            ok ? ++passed : ++failed;
            const double average_depth =
                result.samples != 0
                    ? static_cast<double>(result.path_depth) / result.samples
                    : 0.0;
            std::cout << std::fixed << std::setprecision(6)
                      << "PACKED_TRANSPORT_CHECK id=" << id
                      << " samples=" << result.samples
                      << " errors=" << result.errors
                      << " shadow_rays=" << result.shadow_rays
                      << " average_depth=" << average_depth
                      << " average_luminance=" << result.average_luminance
                      << " checksum=" << std::hex << result.checksum
                      << std::dec << '\n';
        }
        if (selected == 0) {
            throw std::runtime_error("no matching scenes in catalog");
        }
        std::cout << "PACKED_TRANSPORT_CHECK_SUMMARY passed=" << passed
                  << " failed=" << failed << '\n';
        return failed == 0 ? 0 : 1;
    } catch (const std::exception &error) {
        std::cerr << "packed_transport_check: " << error.what() << '\n';
        return 1;
    }
}

#include "device_scene.h"
#include "wavefront_renderer.h"
#include "json.hpp"
#include "packed_transport_core.h"
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
#include <vector>

namespace {

struct Options {
    std::filesystem::path catalog = "assets/scenes/catalog.json";
    std::set<int> scene_ids{1, 7, 23};
    std::uint32_t width = 12;
    std::uint32_t height = 8;
    std::uint32_t spp = 2;
    std::uint32_t max_depth = 8;
    std::uint32_t integrator = 4;
    std::uint32_t seed = 123;
    std::uint32_t batch_size = 31;
    bool all = false;
};

std::set<int> parse_ids(const std::string &value) {
    std::set<int> ids;
    std::istringstream stream(value);
    std::string part;
    while (std::getline(stream, part, ',')) {
        ids.insert(std::stoi(part));
    }
    if (ids.empty()) {
        throw std::runtime_error("--ids requires at least one scene ID");
    }
    return ids;
}

Options parse_options(int argc, char **argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto value = [&]() -> std::string {
            if (++index >= argc) {
                throw std::runtime_error(argument + " requires a value");
            }
            return argv[index];
        };
        if (argument == "--catalog") {
            options.catalog = value();
        } else if (argument == "--ids") {
            options.scene_ids = parse_ids(value());
            options.all = false;
        } else if (argument == "--all") {
            options.all = true;
        } else if (argument == "--width") {
            options.width = static_cast<std::uint32_t>(std::stoul(value()));
        } else if (argument == "--height") {
            options.height = static_cast<std::uint32_t>(std::stoul(value()));
        } else if (argument == "--spp") {
            options.spp = static_cast<std::uint32_t>(std::stoul(value()));
        } else if (argument == "--max-depth") {
            options.max_depth =
                static_cast<std::uint32_t>(std::stoul(value()));
        } else if (argument == "--integrator") {
            options.integrator =
                static_cast<std::uint32_t>(std::stoul(value()));
        } else if (argument == "--seed") {
            options.seed = static_cast<std::uint32_t>(std::stoul(value()));
        } else if (argument == "--batch-size") {
            options.batch_size =
                static_cast<std::uint32_t>(std::stoul(value()));
        } else if (argument == "--help" || argument == "-h") {
            std::cout
                << "Usage: cuda_transport_check [--catalog PATH] [--all] "
                   "[--ids 1,7,...] [--width N] [--height N] [--spp N] "
                   "[--max-depth N] [--integrator 0..4] [--seed N] "
                   "[--batch-size N]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + argument);
        }
    }
    if (options.width == 0 || options.height == 0 || options.spp == 0 ||
        options.max_depth == 0 || options.integrator > 4 ||
        options.batch_size == 0) {
        throw std::runtime_error("invalid CUDA transport check settings");
    }
    return options;
}

nlohmann::json load_json(const std::filesystem::path &path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open catalog: " + path.string());
    }
    nlohmann::json value;
    input >> value;
    return value;
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

cuda_backend::CudaRenderSettings make_settings(const Options &options) {
    cuda_backend::CudaRenderSettings settings;
    settings.transport.policy = integrator_policy(
        integrator_kind_from_id(static_cast<int>(options.integrator)));
    settings.transport.max_depth = options.max_depth;
    settings.width = options.width;
    settings.height = options.height;
    settings.samples_per_pixel = options.spp;
    settings.seed = options.seed;
    settings.batch_size = options.batch_size;
    return settings;
}

cuda_backend::CudaRenderOutput render_cpu(
    const CompiledScene &scene,
    const cuda_backend::CudaRenderSettings &settings) {
    cuda_backend::CudaRenderOutput output;
    const CompiledSceneView view = make_scene_view(scene);
    const std::uint32_t pixel_count = settings.width * settings.height;
    output.film.resize(pixel_count);
    output.stats.sample_count =
        static_cast<std::uint64_t>(pixel_count) *
        settings.samples_per_pixel;
    for (std::uint32_t sample = 0;
         sample < settings.samples_per_pixel; ++sample) {
        for (std::uint32_t pixel = 0; pixel < pixel_count; ++pixel) {
            RNG rng(packed_transport::packed_camera_sample_seed(
                settings.seed, pixel, sample));
            const PackedRay ray =
                packed_transport::generate_packed_camera_ray_core(
                    scene.camera, pixel % settings.width,
                    pixel / settings.width, settings.width, settings.height,
                    rng);
            const PackedTransportResult result =
                packed_transport::trace_packed_path_core(
                    view, ray, settings.transport, rng);
            const std::uint32_t status =
                static_cast<std::uint32_t>(result.status);
            if (status < output.stats.status_counts.size()) {
                ++output.stats.status_counts[status];
            }
            output.stats.traversal_steps += result.traversal_steps;
            output.stats.shadow_rays += result.shadow_rays;
            Float3 radiance = result.radiance;
            if (result.status != PackedTransportStatus::Success ||
                !packed_transport::math::finite(radiance)) {
                radiance = {};
                ++output.stats.invalid_samples;
            }
            cuda_backend::CudaFilmPixel &film = output.film[pixel];
            film.radiance =
                packed_transport::math::add(film.radiance, radiance);
            ++film.sample_count;
        }
    }
    return output;
}

bool nearly_equal(float cpu, float gpu) {
    constexpr float kAbsolute = 2e-3f;
    constexpr float kRelative = 2e-2f;
    return std::abs(cpu - gpu) <=
           kAbsolute + kRelative * std::max(std::abs(cpu), std::abs(gpu));
}

std::size_t compare_film(
    const std::vector<cuda_backend::CudaFilmPixel> &cpu,
    const std::vector<cuda_backend::CudaFilmPixel> &gpu) {
    if (cpu.size() != gpu.size()) {
        return std::max(cpu.size(), gpu.size());
    }
    std::size_t errors = 0;
    for (std::size_t index = 0; index < cpu.size(); ++index) {
        if (cpu[index].sample_count != gpu[index].sample_count ||
            !nearly_equal(cpu[index].radiance.x, gpu[index].radiance.x) ||
            !nearly_equal(cpu[index].radiance.y, gpu[index].radiance.y) ||
            !nearly_equal(cpu[index].radiance.z, gpu[index].radiance.z)) {
            ++errors;
        }
    }
    return errors;
}

bool identical_output(const cuda_backend::CudaRenderOutput &first,
                      const cuda_backend::CudaRenderOutput &second) {
    return first.film.size() == second.film.size() &&
           std::memcmp(first.film.data(), second.film.data(),
                       first.film.size() *
                           sizeof(cuda_backend::CudaFilmPixel)) == 0 &&
           first.stats.status_counts == second.stats.status_counts &&
           first.stats.traversal_steps == second.stats.traversal_steps &&
           first.stats.shadow_rays == second.stats.shadow_rays;
}

bool check_scene(int id, const std::filesystem::path &path,
                 const Options &options) {
    const CompiledScene scene = load_compiled_scene(path.string());
    const cuda_backend::CudaRenderSettings settings =
        make_settings(options);
    const cuda_backend::CudaRenderOutput cpu = render_cpu(scene, settings);
    cuda_backend::DeviceSceneStorage device_scene;
    const cuda_backend::DeviceSceneUploadStats upload =
        device_scene.upload(scene);
    const cuda_backend::CudaRenderOutput gpu =
        cuda_backend::render_wavefront_cuda(device_scene.view(), settings);
    const cuda_backend::CudaRenderOutput repeated =
        cuda_backend::render_wavefront_cuda(device_scene.view(), settings);
    const std::size_t pixel_errors = compare_film(cpu.film, gpu.film);
    const bool deterministic = identical_output(gpu, repeated);
    const bool status_match =
        cpu.stats.status_counts == gpu.stats.status_counts;
    const bool passed = pixel_errors == 0 && deterministic && status_match &&
                        gpu.stats.invalid_samples == 0;
    std::cout << "CUDA_TRANSPORT scene=" << id
              << " pixels=" << settings.width * settings.height
              << " spp=" << settings.samples_per_pixel
              << " depth=" << settings.transport.max_depth
              << " pixel_errors=" << pixel_errors
              << " status_match=" << (status_match ? 1 : 0)
              << " deterministic=" << (deterministic ? 1 : 0)
              << " invalid=" << gpu.stats.invalid_samples
              << " traversal_steps=" << gpu.stats.traversal_steps
              << " shadow_rays=" << gpu.stats.shadow_rays
              << " upload_ms=" << std::fixed << std::setprecision(3)
              << upload.milliseconds
              << " render_ms=" << gpu.stats.milliseconds
              << " workspace=" << gpu.stats.workspace_bytes
              << " result=" << (passed ? "pass" : "fail") << '\n';
    return passed;
}

} // namespace

int main(int argc, char **argv) {
    try {
        std::string reason;
        if (!cuda_backend::cuda_device_available(&reason)) {
            std::cout << "CUDA_TRANSPORT_SKIP reason=" << reason << '\n';
            return 77;
        }
        const Options options = parse_options(argc, argv);
        const nlohmann::json catalog = load_json(options.catalog);
        std::size_t passed = 0;
        std::size_t failed = 0;
        for (const nlohmann::json &entry : catalog.at("scenes")) {
            const int id = entry.at("id").get<int>();
            if (!options.all && options.scene_ids.count(id) == 0) {
                continue;
            }
            const std::filesystem::path path = resolve_scene_path(
                options.catalog, entry.at("path").get<std::string>());
            try {
                if (check_scene(id, path, options)) {
                    ++passed;
                } else {
                    ++failed;
                }
            } catch (const std::exception &error) {
                ++failed;
                std::cerr << "CUDA_TRANSPORT_ERROR scene=" << id
                          << " error=" << error.what() << '\n';
            }
        }
        std::cout << "CUDA_TRANSPORT_SUMMARY passed=" << passed
                  << " failed=" << failed << '\n';
        return failed == 0 && passed != 0 ? 0 : 1;
    } catch (const std::exception &error) {
        std::cerr << "cuda_transport_check: " << error.what() << '\n';
        return 1;
    }
}

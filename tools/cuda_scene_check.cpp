#include "device_buffer.h"
#include "device_scene.h"
#include "intersection_kernels.h"
#include "json.hpp"
#include "flat_intersector.h"
#include "scene_compiler.h"
#include "triangle_scale_fixture.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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
    std::set<int> scene_ids{1, 7, 8, 23, 59, 62, 64};
    int grid_width = 9;
    int grid_height = 7;
    std::uint32_t block_size = 128;
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
        } else if (argument == "--grid-width") {
            options.grid_width = std::stoi(require_value());
        } else if (argument == "--grid-height") {
            options.grid_height = std::stoi(require_value());
        } else if (argument == "--block-size") {
            options.block_size =
                static_cast<std::uint32_t>(std::stoul(require_value()));
        } else if (argument == "--help" || argument == "-h") {
            std::cout
                << "Usage: cuda_scene_check [--catalog PATH] [--all] "
                   "[--ids 1,7,...] [--grid-width N] [--grid-height N] "
                   "[--block-size N]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + argument);
        }
    }
    if (options.grid_width <= 0 || options.grid_height <= 0) {
        throw std::runtime_error("ray grid dimensions must be positive");
    }
    if (options.block_size == 0 || options.block_size > 1024) {
        throw std::runtime_error("block size must be between 1 and 1024");
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
    return std::filesystem::exists(relative) ? relative : scene_path;
}

PackedRay make_camera_ray(const PackedCamera &camera, int x, int y,
                          int width, int height) {
    const float u = (static_cast<float>(x) + 0.37f) / width;
    const float v = (static_cast<float>(y) + 0.61f) / height;
    PackedRay ray;
    ray.origin = camera.origin;
    ray.direction = {
        camera.lower_left_corner.x + u * camera.horizontal.x +
            v * camera.vertical.x - camera.origin.x,
        camera.lower_left_corner.y + u * camera.horizontal.y +
            v * camera.vertical.y - camera.origin.y,
        camera.lower_left_corner.z + u * camera.horizontal.z +
            v * camera.vertical.z - camera.origin.z};
    ray.t_min = 0.001f;
    ray.t_max = 1.0e30f;
    ray.time = camera.time0 + 0.43f * (camera.time1 - camera.time0);
    return ray;
}

bool nearly_equal(float a, float b, float tolerance = 5e-4f) {
    return std::abs(a - b) <= tolerance;
}

struct CheckResult {
    std::size_t rays = 0;
    std::size_t hits = 0;
    std::size_t errors = 0;
    std::size_t bytes = 0;
    float upload_ms = 0.0f;
    float incremental_upload_ms = 0.0f;
    float kernel_ms = 0.0f;
};

CheckResult check_packed_scene(
    const std::string &label, const CompiledScene &packed,
    const std::vector<PackedRay> &rays,
    const std::vector<triangle_scale_fixture::ExpectedHit> *expected,
    const Options &options) {
    const CompiledSceneView host_view = make_scene_view(packed);
    const std::size_t ray_count = rays.size();
    if (expected != nullptr && expected->size() != ray_count) {
        throw std::runtime_error(
            "synthetic ray and expected-result counts differ");
    }

    std::vector<std::uint32_t> initial_rng_states;
    initial_rng_states.reserve(ray_count);
    for (std::size_t index = 0; index < ray_count; ++index) {
        initial_rng_states.push_back(mix_seed(
            1234u, static_cast<std::uint32_t>(index + 1)));
    }

    std::vector<PackedHit> cpu_hits(ray_count);
    std::vector<PackedTraversalStatus> cpu_status(ray_count);
    std::vector<std::uint32_t> cpu_rng_states(ray_count);
    for (std::size_t index = 0; index < ray_count; ++index) {
        RNG rng(initial_rng_states[index]);
        cpu_status[index] = intersect_compiled_scene_status(
            host_view, rays[index], cpu_hits[index], &rng);
        cpu_rng_states[index] = rng.state;
    }

    cuda_backend::DeviceSceneStorage device_scene;
    const cuda_backend::DeviceSceneUploadStats upload =
        device_scene.upload(packed);
    const cuda_backend::DeviceSceneUploadStats incremental_upload =
        device_scene.upload(packed);
    cuda_backend::DeviceBuffer<PackedRay> device_rays;
    cuda_backend::DeviceBuffer<PackedHit> device_hits;
    cuda_backend::DeviceBuffer<PackedTraversalStatus> device_status;
    cuda_backend::DeviceBuffer<std::uint32_t> device_rng_states;
    device_rays.upload(rays);
    device_hits.allocate(ray_count);
    device_status.allocate(ray_count);
    device_rng_states.upload(initial_rng_states);

    const cuda_backend::CudaIntersectionStats kernel =
        cuda_backend::intersect_rays_cuda(
            device_scene.view(), device_rays.data(), device_hits.data(),
            device_status.data(), device_rng_states.data(),
            static_cast<std::uint32_t>(ray_count), options.block_size);

    std::vector<PackedHit> gpu_hits;
    std::vector<PackedTraversalStatus> gpu_status;
    std::vector<std::uint32_t> gpu_rng_states;
    device_hits.download(gpu_hits);
    device_status.download(gpu_status);
    device_rng_states.download(gpu_rng_states);

    CheckResult result;
    result.rays = ray_count;
    result.bytes = upload.bytes;
    result.upload_ms = upload.milliseconds;
    result.incremental_upload_ms = incremental_upload.milliseconds;
    result.kernel_ms = kernel.milliseconds;
    std::size_t reported = 0;
    auto report = [&](std::size_t index, const std::string &message) {
        ++result.errors;
        if (reported++ < 8) {
            std::cerr << "CUDA_RAY_MISMATCH scene=" << label
                      << " ray=" << index << " error=" << message << '\n';
        }
    };

    for (std::size_t index = 0; index < ray_count; ++index) {
        if (expected != nullptr) {
            const triangle_scale_fixture::ExpectedHit &value =
                (*expected)[index];
            const bool cpu_found =
                cpu_status[index] == PackedTraversalStatus::Hit;
            if (cpu_found != value.found) {
                report(index, "host hit/miss differs from expected result");
            } else if (cpu_found) {
                const PackedHit &cpu = cpu_hits[index];
                const float t_tolerance =
                    5e-5f * std::max(1.0f, std::abs(value.t));
                if (!nearly_equal(cpu.t, value.t, t_tolerance) ||
                    !nearly_equal(cpu.barycentric_u,
                                  value.barycentric_u, 2e-5f) ||
                    !nearly_equal(cpu.barycentric_v,
                                  value.barycentric_v, 2e-5f)) {
                    report(index, "host hit differs from expected values");
                }
                if (cpu.instance_id != 0 || cpu.primitive_id != 0) {
                    report(index,
                           "host primitive or instance ID is unexpected");
                }
            }
        }
        if (cpu_status[index] != gpu_status[index]) {
            report(index, "traversal status differs");
            continue;
        }
        if (cpu_rng_states[index] != gpu_rng_states[index]) {
            report(index, "RNG state differs");
        }
        if (cpu_status[index] != PackedTraversalStatus::Hit) {
            continue;
        }
        ++result.hits;
        const PackedHit &cpu = cpu_hits[index];
        const PackedHit &gpu = gpu_hits[index];
        if (!nearly_equal(cpu.t, gpu.t)) {
            std::ostringstream message;
            message << "hit distance differs cpu=" << cpu.t
                    << " gpu=" << gpu.t << " cpu_flags=" << cpu.flags
                    << " gpu_flags=" << gpu.flags;
            report(index, message.str());
        }
        if (!nearly_equal(cpu.barycentric_u, gpu.barycentric_u) ||
            !nearly_equal(cpu.barycentric_v, gpu.barycentric_v)) {
            std::ostringstream message;
            message << "barycentrics differ cpu=(" << cpu.barycentric_u
                    << ',' << cpu.barycentric_v << ") gpu=("
                    << gpu.barycentric_u << ',' << gpu.barycentric_v << ')';
            report(index, message.str());
        }
        if (cpu.instance_id != gpu.instance_id) {
            report(index, "instance ID differs");
        }
        if (cpu.primitive_id != gpu.primitive_id) {
            report(index, "primitive ID differs");
        }
        if (cpu.flags != gpu.flags) {
            report(index, "hit flags differ");
        }
    }
    return result;
}

CheckResult check_scene(const std::filesystem::path &path,
                        const Options &options) {
    const CompiledScene packed = load_compiled_scene(path.string());
    std::vector<PackedRay> rays;
    rays.reserve(static_cast<std::size_t>(options.grid_width) *
                 options.grid_height);
    for (int y = 0; y < options.grid_height; ++y) {
        for (int x = 0; x < options.grid_width; ++x) {
            rays.push_back(make_camera_ray(packed.camera, x, y,
                                           options.grid_width,
                                           options.grid_height));
        }
    }
    return check_packed_scene(path.string(), packed, rays, nullptr, options);
}

CheckResult check_triangle_scale_fixture(const Options &options) {
    const triangle_scale_fixture::Fixture fixture =
        triangle_scale_fixture::make_fixture();
    return check_packed_scene("synthetic_scaled_triangle", fixture.packed,
                              fixture.rays, &fixture.expected, options);
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Options options = parse_options(argc, argv);
        std::string unavailable_reason;
        if (!cuda_backend::cuda_device_available(&unavailable_reason)) {
            std::cout << "CUDA_SCENE_CHECK_SKIP reason=\""
                      << unavailable_reason << "\"\n";
            return 77;
        }
        const cuda_backend::DeviceMemoryInfo memory =
            cuda_backend::query_device_memory();
        std::cout << "CUDA_DEVICE name=\"" << cuda_backend::cuda_device_name()
                  << "\" total_bytes=" << memory.total_bytes
                  << " free_bytes=" << memory.free_bytes << '\n';

        const nlohmann::json catalog = load_json(options.catalog);
        std::size_t passed = 0;
        std::size_t failed = 0;
        std::size_t selected = 0;
        const CheckResult scale_result =
            check_triangle_scale_fixture(options);
        const bool scale_ok = scale_result.errors == 0;
        std::cout << "CUDA_TRIANGLE_SCALE_CHECK rays=" << scale_result.rays
                  << " hits=" << scale_result.hits
                  << " errors=" << scale_result.errors
                  << " bytes=" << scale_result.bytes
                  << " upload_ms=" << scale_result.upload_ms
                  << " incremental_upload_ms="
                  << scale_result.incremental_upload_ms
                  << " kernel_ms=" << scale_result.kernel_ms << '\n';
        for (const nlohmann::json &entry : catalog.at("scenes")) {
            const int id = entry.at("id").get<int>();
            if (!options.all && options.scene_ids.count(id) == 0) {
                continue;
            }
            ++selected;
            const std::filesystem::path path = resolve_scene_path(
                options.catalog, entry.at("path").get<std::string>());
            const CheckResult result = check_scene(path, options);
            const bool ok = result.errors == 0;
            ok ? ++passed : ++failed;
            const double rays_per_second =
                result.kernel_ms > 0.0f
                    ? result.rays * 1000.0 / result.kernel_ms
                    : 0.0;
            std::cout << std::fixed << std::setprecision(3)
                      << "CUDA_SCENE_CHECK id=" << id
                      << " rays=" << result.rays << " hits=" << result.hits
                      << " errors=" << result.errors
                      << " bytes=" << result.bytes
                      << " upload_ms=" << result.upload_ms
                      << " kernel_ms=" << result.kernel_ms
                      << " rays_per_second=" << std::setprecision(0)
                      << rays_per_second << '\n';
        }
        if (selected == 0) {
            throw std::runtime_error("no matching scenes in catalog");
        }
        std::cout << "CUDA_SCENE_CHECK_SUMMARY passed=" << passed
                  << " failed=" << failed << '\n';
        return failed == 0 && scale_ok ? 0 : 1;
    } catch (const std::exception &error) {
        std::cerr << "cuda_scene_check: " << error.what() << '\n';
        return 1;
    }
}

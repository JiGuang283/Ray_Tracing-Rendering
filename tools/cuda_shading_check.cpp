#include "cuda/device_buffer.h"
#include "cuda/device_scene.h"
#include "cuda/intersection_kernels.h"
#include "cuda/shading_kernels.h"
#include "external/json.hpp"
#include "render_data/flat_intersector.h"
#include "render_data/packed_material.h"
#include "render_data/scene_compiler.h"

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

struct CheckResult {
    std::size_t rays = 0;
    std::size_t hits = 0;
    std::size_t materials = 0;
    std::size_t errors = 0;
    std::size_t bytes = 0;
    std::uint32_t max_texture_stack = 0;
    float upload_ms = 0.0f;
    float intersection_ms = 0.0f;
    float reconstruction_ms = 0.0f;
    float material_ms = 0.0f;
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
                << "Usage: cuda_shading_check [--catalog PATH] [--all] "
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

PackedSurfaceInteraction synthetic_surface(std::uint32_t material_id) {
    const float phase = static_cast<float>(material_id % 17u) / 17.0f;
    PackedSurfaceInteraction surface;
    surface.position = {0.25f + phase, -0.5f + 0.3f * phase,
                        1.0f - 0.2f * phase};
    surface.material_id = material_id;
    surface.geometric_normal = {0.0f, 0.0f, 1.0f};
    surface.shading_normal = {0.0499277f, -0.0249639f, 0.998553f};
    surface.dpdu = {1.0f, 0.0f, 0.0f};
    surface.dpdv = {0.0f, 1.0f, 0.0f};
    surface.uv = {0.13f + 0.71f * phase, 0.83f - 0.59f * phase};
    surface.vertex_color = {0.8f, 0.65f, 0.45f, 0.9f};
    surface.vertex_alpha = 0.9f;
    surface.flags = PACKED_HIT_FRONT_FACE;
    return surface;
}

bool nearly_equal(float a, float b, float absolute_tolerance = 5e-5f,
                  float relative_tolerance = 5e-5f) {
    if (!std::isfinite(a) || !std::isfinite(b)) {
        return a == b;
    }
    return std::abs(a - b) <=
           absolute_tolerance +
               relative_tolerance * std::max(std::abs(a), std::abs(b));
}

bool nearly_equal(Float2 a, Float2 b, float tolerance = 5e-4f) {
    return nearly_equal(a.x, b.x, tolerance, tolerance) &&
           nearly_equal(a.y, b.y, tolerance, tolerance);
}

bool nearly_equal(Float3 a, Float3 b, float tolerance = 5e-4f) {
    return nearly_equal(a.x, b.x, tolerance, tolerance) &&
           nearly_equal(a.y, b.y, tolerance, tolerance) &&
           nearly_equal(a.z, b.z, tolerance, tolerance);
}

bool nearly_equal(Float4 a, Float4 b, float tolerance = 5e-4f) {
    return nearly_equal(a.x, b.x, tolerance, tolerance) &&
           nearly_equal(a.y, b.y, tolerance, tolerance) &&
           nearly_equal(a.z, b.z, tolerance, tolerance) &&
           nearly_equal(a.w, b.w, tolerance, tolerance);
}

float normalized_dot(Float3 a, Float3 b) {
    const float a_length =
        std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
    const float b_length =
        std::sqrt(b.x * b.x + b.y * b.y + b.z * b.z);
    if (!(a_length > 0.0f) || !(b_length > 0.0f)) {
        return a_length == b_length ? 1.0f : 0.0f;
    }
    return (a.x * b.x + a.y * b.y + a.z * b.z) /
           (a_length * b_length);
}

std::string compare_surface(const PackedSurfaceInteraction &cpu,
                            const PackedSurfaceInteraction &gpu) {
    if (cpu.material_id != gpu.material_id ||
        cpu.instance_id != gpu.instance_id ||
        cpu.primitive_id != gpu.primitive_id || cpu.flags != gpu.flags) {
        return "surface identifiers differ";
    }
    if (!nearly_equal(cpu.position, gpu.position)) {
        return "surface position differs";
    }
    if (normalized_dot(cpu.geometric_normal, gpu.geometric_normal) <
            0.999f ||
        normalized_dot(cpu.shading_normal, gpu.shading_normal) < 0.999f) {
        return "surface normal differs";
    }
    if (!nearly_equal(cpu.dpdu, gpu.dpdu, 2e-3f) ||
        !nearly_equal(cpu.dpdv, gpu.dpdv, 2e-3f)) {
        return "surface derivatives differ";
    }
    if (!nearly_equal(cpu.uv, gpu.uv) ||
        !nearly_equal(cpu.vertex_color, gpu.vertex_color) ||
        !nearly_equal(cpu.vertex_alpha, gpu.vertex_alpha)) {
        return "surface attributes differ";
    }
    return {};
}

std::string compare_material_output(const PackedMaterialOutput &cpu,
                                    const PackedMaterialOutput &gpu) {
    if (cpu.closure_count != gpu.closure_count) {
        return "closure count differs";
    }
    if (normalized_dot(cpu.frame.normal, gpu.frame.normal) < 0.999f ||
        normalized_dot(cpu.frame.tangent, gpu.frame.tangent) < 0.999f ||
        normalized_dot(cpu.geometry_normal, gpu.geometry_normal) < 0.999f) {
        return "material shading frame differs";
    }
    if (!nearly_equal(cpu.frame.handedness, gpu.frame.handedness) ||
        !nearly_equal(cpu.emission, gpu.emission) ||
        !nearly_equal(cpu.opacity, gpu.opacity)) {
        return "material scalar or emission output differs";
    }
    for (std::uint32_t index = 0; index < cpu.closure_count; ++index) {
        const PackedClosure &cpu_closure = cpu.closures[index];
        const PackedClosure &gpu_closure = gpu.closures[index];
        if (cpu_closure.type != gpu_closure.type ||
            cpu_closure.flags != gpu_closure.flags) {
            return "closure type or flags differ";
        }
        if (!nearly_equal(cpu_closure.parameters,
                          gpu_closure.parameters) ||
            !nearly_equal(cpu_closure.contribution_weight,
                          gpu_closure.contribution_weight) ||
            !nearly_equal(cpu_closure.sample_weight,
                          gpu_closure.sample_weight)) {
            return "closure parameters differ";
        }
    }
    return {};
}

void evaluate_cpu_materials(
    const CompiledSceneView &view,
    const std::vector<PackedSurfaceInteraction> &surfaces,
    const std::vector<PackedShadingStatus> &input_status,
    std::vector<PackedMaterialOutput> &outputs,
    std::vector<PackedShadingStatus> &statuses,
    std::vector<std::uint32_t> &stack_usage) {
    outputs.resize(surfaces.size());
    statuses.resize(surfaces.size());
    stack_usage.assign(surfaces.size(), 0);
    for (std::size_t index = 0; index < surfaces.size(); ++index) {
        statuses[index] = input_status[index];
        if (statuses[index] == PackedShadingStatus::Success) {
            statuses[index] = evaluate_packed_material_status(
                view, surfaces[index].material_id, surfaces[index],
                outputs[index], &stack_usage[index]);
        }
    }
}

CheckResult check_scene(const std::filesystem::path &path,
                        const Options &options) {
    const CompiledScene packed = load_compiled_scene(path.string());
    const CompiledSceneView host_view = make_scene_view(packed);
    const std::size_t ray_count =
        static_cast<std::size_t>(options.grid_width) * options.grid_height;

    std::vector<PackedRay> rays;
    std::vector<std::uint32_t> initial_rng_states;
    rays.reserve(ray_count);
    initial_rng_states.reserve(ray_count);
    for (int y = 0; y < options.grid_height; ++y) {
        for (int x = 0; x < options.grid_width; ++x) {
            rays.push_back(make_camera_ray(packed.camera, x, y,
                                           options.grid_width,
                                           options.grid_height));
            initial_rng_states.push_back(mix_seed(
                1234u, static_cast<std::uint32_t>(rays.size())));
        }
    }

    std::vector<PackedHit> cpu_hits(ray_count);
    std::vector<PackedTraversalStatus> cpu_traversal(ray_count);
    std::vector<PackedSurfaceInteraction> cpu_surfaces(ray_count);
    std::vector<PackedShadingStatus> cpu_reconstruction(
        ray_count, PackedShadingStatus::Miss);
    for (std::size_t index = 0; index < ray_count; ++index) {
        RNG rng(initial_rng_states[index]);
        cpu_traversal[index] = intersect_compiled_scene_status(
            host_view, rays[index], cpu_hits[index], &rng);
        if (cpu_traversal[index] == PackedTraversalStatus::Hit) {
            cpu_reconstruction[index] = reconstruct_compiled_hit_status(
                host_view, rays[index], cpu_hits[index],
                cpu_surfaces[index]);
        } else if (cpu_traversal[index] != PackedTraversalStatus::Miss) {
            cpu_reconstruction[index] = PackedShadingStatus::InvalidInput;
        }
    }
    std::vector<PackedMaterialOutput> cpu_outputs;
    std::vector<PackedShadingStatus> cpu_material_status;
    std::vector<std::uint32_t> cpu_stack_usage;
    evaluate_cpu_materials(host_view, cpu_surfaces, cpu_reconstruction,
                           cpu_outputs, cpu_material_status,
                           cpu_stack_usage);

    cuda_backend::DeviceSceneStorage device_scene;
    const cuda_backend::DeviceSceneUploadStats upload =
        device_scene.upload(packed);
    cuda_backend::DeviceBuffer<PackedRay> device_rays;
    cuda_backend::DeviceBuffer<PackedHit> device_hits;
    cuda_backend::DeviceBuffer<PackedTraversalStatus> device_traversal;
    cuda_backend::DeviceBuffer<std::uint32_t> device_rng_states;
    cuda_backend::DeviceBuffer<PackedSurfaceInteraction> device_surfaces;
    cuda_backend::DeviceBuffer<PackedShadingStatus> device_reconstruction;
    cuda_backend::DeviceBuffer<PackedMaterialOutput> device_outputs;
    cuda_backend::DeviceBuffer<PackedShadingStatus> device_material_status;
    cuda_backend::DeviceBuffer<std::uint32_t> device_stack_usage;
    device_rays.upload(rays);
    device_hits.allocate(ray_count);
    device_traversal.allocate(ray_count);
    device_rng_states.upload(initial_rng_states);
    device_surfaces.allocate(ray_count);
    device_reconstruction.allocate(ray_count);
    device_outputs.allocate(ray_count);
    device_material_status.allocate(ray_count);
    device_stack_usage.allocate(ray_count);

    const cuda_backend::CudaIntersectionStats intersection =
        cuda_backend::intersect_rays_cuda(
            device_scene.view(), device_rays.data(), device_hits.data(),
            device_traversal.data(), device_rng_states.data(),
            static_cast<std::uint32_t>(ray_count), options.block_size);
    const cuda_backend::CudaShadingStageStats reconstruction =
        cuda_backend::reconstruct_hits_cuda(
            device_scene.view(), device_rays.data(), device_hits.data(),
            device_traversal.data(), device_surfaces.data(),
            device_reconstruction.data(),
            static_cast<std::uint32_t>(ray_count), options.block_size);
    const cuda_backend::CudaShadingStageStats material =
        cuda_backend::evaluate_materials_cuda(
            device_scene.view(), device_surfaces.data(),
            device_reconstruction.data(), device_outputs.data(),
            device_material_status.data(), device_stack_usage.data(),
            static_cast<std::uint32_t>(ray_count), options.block_size);

    std::vector<PackedTraversalStatus> gpu_traversal;
    std::vector<PackedSurfaceInteraction> gpu_surfaces;
    std::vector<PackedShadingStatus> gpu_reconstruction;
    std::vector<PackedMaterialOutput> gpu_outputs;
    std::vector<PackedShadingStatus> gpu_material_status;
    std::vector<std::uint32_t> gpu_stack_usage;
    device_traversal.download(gpu_traversal);
    device_surfaces.download(gpu_surfaces);
    device_reconstruction.download(gpu_reconstruction);
    device_outputs.download(gpu_outputs);
    device_material_status.download(gpu_material_status);
    device_stack_usage.download(gpu_stack_usage);

    CheckResult result;
    result.rays = ray_count;
    result.materials = packed.materials.size();
    result.bytes = upload.bytes;
    result.upload_ms = upload.milliseconds;
    result.intersection_ms = intersection.milliseconds;
    result.reconstruction_ms = reconstruction.milliseconds;
    result.material_ms = material.milliseconds;
    std::size_t reported = 0;
    auto report = [&](const std::string &kind, std::size_t index,
                      const std::string &message) {
        ++result.errors;
        if (reported++ < 12) {
            std::cerr << "CUDA_SHADING_MISMATCH scene=" << path.string()
                      << " kind=" << kind << " index=" << index
                      << " error=" << message << '\n';
        }
    };

    for (std::size_t index = 0; index < ray_count; ++index) {
        if (cpu_traversal[index] == PackedTraversalStatus::Hit &&
            cpu_reconstruction[index] != PackedShadingStatus::Success) {
            report("ray", index, "host reconstruction failed");
        }
        if (cpu_traversal[index] != gpu_traversal[index]) {
            report("ray", index, "traversal status differs");
            continue;
        }
        if (cpu_reconstruction[index] != gpu_reconstruction[index]) {
            report("ray", index, "reconstruction status differs");
            continue;
        }
        if (cpu_reconstruction[index] == PackedShadingStatus::Success) {
            ++result.hits;
            const std::string mismatch =
                compare_surface(cpu_surfaces[index], gpu_surfaces[index]);
            if (!mismatch.empty()) {
                report("ray", index, mismatch);
            }
        }
        if (cpu_material_status[index] != gpu_material_status[index]) {
            report("ray", index, "material status differs");
            continue;
        }
        if (cpu_reconstruction[index] == PackedShadingStatus::Success &&
            cpu_material_status[index] != PackedShadingStatus::Success) {
            report("ray", index, "host material evaluation failed");
            continue;
        }
        if (cpu_stack_usage[index] != gpu_stack_usage[index]) {
            report("ray", index, "texture stack usage differs");
        }
        result.max_texture_stack =
            std::max(result.max_texture_stack, gpu_stack_usage[index]);
        if (cpu_material_status[index] == PackedShadingStatus::Success) {
            const std::string mismatch = compare_material_output(
                cpu_outputs[index], gpu_outputs[index]);
            if (!mismatch.empty()) {
                report("ray", index, mismatch);
            }
        }
    }

    std::vector<PackedSurfaceInteraction> material_surfaces;
    std::vector<PackedShadingStatus> material_inputs(
        packed.materials.size(), PackedShadingStatus::Success);
    material_surfaces.reserve(packed.materials.size());
    for (std::uint32_t index = 0; index < packed.materials.size(); ++index) {
        material_surfaces.push_back(synthetic_surface(index));
    }
    std::vector<PackedMaterialOutput> cpu_material_outputs;
    std::vector<PackedShadingStatus> cpu_material_statuses;
    std::vector<std::uint32_t> cpu_material_stacks;
    evaluate_cpu_materials(host_view, material_surfaces, material_inputs,
                           cpu_material_outputs, cpu_material_statuses,
                           cpu_material_stacks);

    cuda_backend::DeviceBuffer<PackedSurfaceInteraction>
        device_material_surfaces;
    cuda_backend::DeviceBuffer<PackedShadingStatus> device_material_inputs;
    cuda_backend::DeviceBuffer<PackedMaterialOutput>
        device_all_material_outputs;
    cuda_backend::DeviceBuffer<PackedShadingStatus>
        device_all_material_statuses;
    cuda_backend::DeviceBuffer<std::uint32_t> device_material_stacks;
    device_material_surfaces.upload(material_surfaces);
    device_material_inputs.upload(material_inputs);
    device_all_material_outputs.allocate(packed.materials.size());
    device_all_material_statuses.allocate(packed.materials.size());
    device_material_stacks.allocate(packed.materials.size());
    const cuda_backend::CudaShadingStageStats all_materials =
        cuda_backend::evaluate_materials_cuda(
            device_scene.view(), device_material_surfaces.data(),
            device_material_inputs.data(),
            device_all_material_outputs.data(),
            device_all_material_statuses.data(),
            device_material_stacks.data(),
            static_cast<std::uint32_t>(packed.materials.size()),
            options.block_size);
    result.material_ms += all_materials.milliseconds;

    std::vector<PackedMaterialOutput> gpu_material_outputs;
    std::vector<PackedShadingStatus> gpu_material_statuses;
    std::vector<std::uint32_t> gpu_material_stacks;
    device_all_material_outputs.download(gpu_material_outputs);
    device_all_material_statuses.download(gpu_material_statuses);
    device_material_stacks.download(gpu_material_stacks);
    for (std::size_t index = 0; index < packed.materials.size(); ++index) {
        if (cpu_material_statuses[index] !=
            PackedShadingStatus::Success) {
            report("material", index, "host material evaluation failed");
            continue;
        }
        if (cpu_material_statuses[index] != gpu_material_statuses[index]) {
            report("material", index, "material status differs");
            continue;
        }
        if (cpu_material_stacks[index] != gpu_material_stacks[index]) {
            report("material", index, "texture stack usage differs");
        }
        result.max_texture_stack =
            std::max(result.max_texture_stack, gpu_material_stacks[index]);
        if (cpu_material_statuses[index] == PackedShadingStatus::Success) {
            const std::string mismatch = compare_material_output(
                cpu_material_outputs[index], gpu_material_outputs[index]);
            if (!mismatch.empty()) {
                report("material", index, mismatch);
            }
        }
    }
    return result;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Options options = parse_options(argc, argv);
        std::string unavailable_reason;
        if (!cuda_backend::cuda_device_available(&unavailable_reason)) {
            std::cout << "CUDA_SHADING_CHECK_SKIP reason=\""
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
            std::cout << std::fixed << std::setprecision(3)
                      << "CUDA_SHADING_CHECK id=" << id
                      << " rays=" << result.rays << " hits=" << result.hits
                      << " materials=" << result.materials
                      << " errors=" << result.errors
                      << " max_texture_stack=" << result.max_texture_stack
                      << " bytes=" << result.bytes
                      << " upload_ms=" << result.upload_ms
                      << " intersection_ms=" << result.intersection_ms
                      << " reconstruction_ms=" << result.reconstruction_ms
                      << " material_ms=" << result.material_ms << '\n';
        }
        if (selected == 0) {
            throw std::runtime_error("no matching scenes in catalog");
        }
        std::cout << "CUDA_SHADING_CHECK_SUMMARY passed=" << passed
                  << " failed=" << failed << '\n';
        return failed == 0 ? 0 : 1;
    } catch (const std::exception &error) {
        std::cerr << "cuda_shading_check: " << error.what() << '\n';
        return 1;
    }
}

#include "device_buffer.h"
#include "device_scene.h"
#include "light_kernels.h"
#include "json.hpp"
#include "packed_light.h"
#include "scene_compiler.h"

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
    std::set<int> scene_ids{7, 8, 11, 23, 59, 62, 64};
    std::uint32_t block_size = 128;
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
        } else if (argument == "--block-size") {
            options.block_size =
                static_cast<std::uint32_t>(std::stoul(require_value()));
        } else if (argument == "--help" || argument == "-h") {
            std::cout << "Usage: cuda_light_check [--catalog PATH] [--all] "
                         "[--ids 1,7,...] [--block-size N]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + argument);
        }
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

bool nearly_equal(float a, float b, float tolerance = 5e-3f) {
    if (!std::isfinite(a) || !std::isfinite(b)) {
        return a == b;
    }
    return std::abs(a - b) <=
           tolerance + tolerance * std::max(std::abs(a), std::abs(b));
}

bool nearly_equal(Float3 a, Float3 b, float tolerance = 5e-3f) {
    return nearly_equal(a.x, b.x, tolerance) &&
           nearly_equal(a.y, b.y, tolerance) &&
           nearly_equal(a.z, b.z, tolerance);
}

std::string compare_sample(const PackedLightSample &cpu,
                           const PackedLightSample &gpu) {
    if (cpu.light_id != gpu.light_id || cpu.flags != gpu.flags ||
        cpu.element_id != gpu.element_id) {
        return "light sample identifiers differ";
    }
    if (!nearly_equal(cpu.wi, gpu.wi) ||
        !nearly_equal(cpu.radiance, gpu.radiance) ||
        !nearly_equal(cpu.distance, gpu.distance) ||
        !nearly_equal(cpu.pdf, gpu.pdf)) {
        return "light sample values differ";
    }
    return {};
}

struct CheckResult {
    std::size_t lights = 0;
    std::size_t queries = 0;
    std::size_t selection_queries = 0;
    std::size_t errors = 0;
    std::size_t bytes = 0;
    float upload_ms = 0.0f;
    float light_ms = 0.0f;
    float selection_ms = 0.0f;
};

CheckResult check_scene(const std::filesystem::path &path,
                        const Options &options) {
    const CompiledScene packed = load_compiled_scene(path.string());
    const CompiledSceneView host_view = make_scene_view(packed);
    constexpr std::uint32_t kQueriesPerLight = 4;
    std::vector<std::uint32_t> light_ids;
    std::vector<Float3> origins;
    std::vector<Float2> random_values;
    for (std::uint32_t light_id = 0; light_id < packed.lights.size();
         ++light_id) {
        for (std::uint32_t query = 0; query < kQueriesPerLight; ++query) {
            light_ids.push_back(light_id);
            origins.push_back(
                {packed.camera.origin.x + 0.37f * query,
                 packed.camera.origin.y - 0.19f * query,
                 packed.camera.origin.z + 0.11f * query});
            random_values.push_back(
                {0.11f + 0.19f * query, 0.83f - 0.17f * query});
        }
    }

    std::vector<PackedLightSample> cpu_samples(light_ids.size());
    std::vector<PackedLightStatus> cpu_statuses(light_ids.size());
    std::vector<float> cpu_pdfs(light_ids.size(), 0.0f);
    std::vector<PackedLightStatus> cpu_pdf_statuses(light_ids.size());
    for (std::size_t index = 0; index < light_ids.size(); ++index) {
        cpu_statuses[index] = sample_packed_light(
            host_view, light_ids[index], origins[index], random_values[index],
            cpu_samples[index]);
        const Float3 direction =
            cpu_statuses[index] == PackedLightStatus::Success
                ? cpu_samples[index].wi
                : Float3{0.381f, 0.889f, -0.254f};
        cpu_pdf_statuses[index] = evaluate_packed_light_pdf(
            host_view, light_ids[index], origins[index], direction,
            cpu_pdfs[index]);
    }

    constexpr std::uint32_t kSelectionQueries = 128;
    const std::uint32_t selection_count =
        packed.non_delta_light_indices.empty() ? 0 : kSelectionQueries;
    std::vector<Float3> selection_origins(selection_count,
                                          packed.camera.origin);
    std::vector<std::uint32_t> initial_rng_states(selection_count);
    std::vector<std::uint32_t> cpu_final_rng_states(selection_count);
    std::vector<SelectedPackedLightSample> cpu_selected(selection_count);
    std::vector<PackedLightStatus> cpu_selection_status(selection_count);
    for (std::uint32_t index = 0; index < selection_count; ++index) {
        initial_rng_states[index] = mix_seed(0x4c494748u, index + 1);
        RNG rng(initial_rng_states[index]);
        cpu_selection_status[index] = sample_packed_non_delta_light(
            host_view, selection_origins[index], rng, cpu_selected[index]);
        cpu_final_rng_states[index] = rng.state;
    }

    cuda_backend::DeviceSceneStorage device_scene;
    const cuda_backend::DeviceSceneUploadStats upload =
        device_scene.upload(packed);
    cuda_backend::DeviceBuffer<std::uint32_t> device_light_ids;
    cuda_backend::DeviceBuffer<Float3> device_origins;
    cuda_backend::DeviceBuffer<Float2> device_random_values;
    cuda_backend::DeviceBuffer<PackedLightSample> device_samples;
    cuda_backend::DeviceBuffer<PackedLightStatus> device_statuses;
    cuda_backend::DeviceBuffer<float> device_pdfs;
    cuda_backend::DeviceBuffer<PackedLightStatus> device_pdf_statuses;
    device_light_ids.upload(light_ids);
    device_origins.upload(origins);
    device_random_values.upload(random_values);
    device_samples.allocate(light_ids.size());
    device_statuses.allocate(light_ids.size());
    device_pdfs.allocate(light_ids.size());
    device_pdf_statuses.allocate(light_ids.size());
    const cuda_backend::CudaLightStageStats light_stats =
        cuda_backend::evaluate_lights_cuda(
            device_scene.view(), device_light_ids.data(),
            device_origins.data(), device_random_values.data(),
            device_samples.data(), device_statuses.data(),
            device_pdfs.data(), device_pdf_statuses.data(),
            static_cast<std::uint32_t>(light_ids.size()), options.block_size);

    std::vector<PackedLightSample> gpu_samples;
    std::vector<PackedLightStatus> gpu_statuses;
    std::vector<float> gpu_pdfs;
    std::vector<PackedLightStatus> gpu_pdf_statuses;
    device_samples.download(gpu_samples);
    device_statuses.download(gpu_statuses);
    device_pdfs.download(gpu_pdfs);
    device_pdf_statuses.download(gpu_pdf_statuses);

    cuda_backend::DeviceBuffer<Float3> device_selection_origins;
    cuda_backend::DeviceBuffer<std::uint32_t> device_rng_states;
    cuda_backend::DeviceBuffer<SelectedPackedLightSample> device_selected;
    cuda_backend::DeviceBuffer<PackedLightStatus> device_selection_status;
    device_selection_origins.upload(selection_origins);
    device_rng_states.upload(initial_rng_states);
    device_selected.allocate(selection_count);
    device_selection_status.allocate(selection_count);
    const cuda_backend::CudaLightStageStats selection_stats =
        cuda_backend::sample_non_delta_lights_cuda(
            device_scene.view(), device_selection_origins.data(),
            device_rng_states.data(), device_selected.data(),
            device_selection_status.data(), selection_count,
            options.block_size);
    std::vector<std::uint32_t> gpu_final_rng_states;
    std::vector<SelectedPackedLightSample> gpu_selected;
    std::vector<PackedLightStatus> gpu_selection_status;
    device_rng_states.download(gpu_final_rng_states);
    device_selected.download(gpu_selected);
    device_selection_status.download(gpu_selection_status);

    CheckResult result;
    result.lights = packed.lights.size();
    result.queries = light_ids.size();
    result.selection_queries = selection_count;
    result.bytes = upload.bytes;
    result.upload_ms = upload.milliseconds;
    result.light_ms = light_stats.milliseconds;
    result.selection_ms = selection_stats.milliseconds;
    std::size_t reported = 0;
    auto report = [&](const std::string &kind, std::size_t index,
                      const std::string &message) {
        ++result.errors;
        if (reported++ < 12) {
            std::cerr << "CUDA_LIGHT_MISMATCH scene=" << path.string()
                      << " kind=" << kind << " index=" << index
                      << " error=" << message << '\n';
        }
    };
    for (std::size_t index = 0; index < light_ids.size(); ++index) {
        if (cpu_statuses[index] != gpu_statuses[index]) {
            report("sample", index, "sample status differs");
            continue;
        }
        if (cpu_pdf_statuses[index] != gpu_pdf_statuses[index] ||
            !nearly_equal(cpu_pdfs[index], gpu_pdfs[index])) {
            report("sample", index, "light PDF differs");
        }
        const std::string mismatch =
            compare_sample(cpu_samples[index], gpu_samples[index]);
        if (!mismatch.empty()) {
            report("sample", index, mismatch);
        }
    }
    for (std::size_t index = 0; index < selection_count; ++index) {
        if (cpu_selection_status[index] != gpu_selection_status[index]) {
            report("selection", index, "selection status differs");
            continue;
        }
        if (cpu_final_rng_states[index] != gpu_final_rng_states[index]) {
            report("selection", index, "selection RNG state differs");
        }
        if (cpu_selected[index].selection_index !=
                gpu_selected[index].selection_index ||
            !nearly_equal(cpu_selected[index].selection_probability,
                          gpu_selected[index].selection_probability,
                          1e-6f)) {
            report("selection", index, "selected light differs");
        }
        const std::string mismatch = compare_sample(
            cpu_selected[index].sample, gpu_selected[index].sample);
        if (!mismatch.empty()) {
            report("selection", index, mismatch);
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
            std::cout << "CUDA_LIGHT_CHECK_SKIP reason=\""
                      << unavailable_reason << "\"\n";
            return 77;
        }
        std::cout << "CUDA_DEVICE name=\"" << cuda_backend::cuda_device_name()
                  << "\"\n";
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
            const CheckResult result = check_scene(path, options);
            const bool ok = result.errors == 0;
            ok ? ++passed : ++failed;
            std::cout << std::fixed << std::setprecision(3)
                      << "CUDA_LIGHT_CHECK id=" << id
                      << " lights=" << result.lights
                      << " queries=" << result.queries
                      << " selection_queries=" << result.selection_queries
                      << " errors=" << result.errors
                      << " bytes=" << result.bytes
                      << " upload_ms=" << result.upload_ms
                      << " light_ms=" << result.light_ms
                      << " selection_ms=" << result.selection_ms << '\n';
        }
        if (selected == 0) {
            throw std::runtime_error("no matching scenes in catalog");
        }
        std::cout << "CUDA_LIGHT_CHECK_SUMMARY passed=" << passed
                  << " failed=" << failed << '\n';
        return failed == 0 ? 0 : 1;
    } catch (const std::exception &error) {
        std::cerr << "cuda_light_check: " << error.what() << '\n';
        return 1;
    }
}

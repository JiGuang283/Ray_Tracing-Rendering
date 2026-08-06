#include "app_options.h"

#include <cstdlib>
#include <cmath>
#include <cerrno>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

bool parse_int_arg(const char *value, int &out) {
    if (value == nullptr) {
        return false;
    }
    char *end = nullptr;
    errno = 0;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || errno == ERANGE ||
        parsed < std::numeric_limits<int>::min() ||
        parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    out = static_cast<int>(parsed);
    return true;
}

bool parse_double_arg(const char *value, double &out) {
    if (value == nullptr) {
        return false;
    }
    char *end = nullptr;
    const double parsed = std::strtod(value, &end);
    if (end == value || *end != '\0' || !std::isfinite(parsed)) {
        return false;
    }
    out = parsed;
    return true;
}

} // namespace

void print_usage() {
    std::cerr
        << "Usage: ./CGAssignment4 [scene] [integrator] [options]\n"
        << "Options:\n"
        << "  --bench              Run headless benchmark mode\n"
        << "  --scene-file PATH    Load scene from a JSON file\n"
        << "  --integrator N       Select integrator for --scene-file mode\n"
        << "  --backend cpu|cuda   Select render backend (default cpu)\n"
        << "  --runs N             Benchmark run count\n"
        << "  --width N            Override image width\n"
        << "  --spp N              Override samples per pixel\n"
        << "  --max-depth N        Override integrator max depth\n"
        << "  --seed N             Set render and scene seed (default 1337)\n"
        << "  --threads N          Set render worker count (default hardware)\n"
        << "  --sample-clamp N     Clamp camera-sample luminance (0 disables)\n"
        << "  --cuda-batch-size N  Override CUDA active-path batch size\n"
        << "  --restir-light-candidates N  Initial DI candidates per pixel\n"
        << "  --restir-spatial-neighbors N Spatial neighbors (max 64)\n"
        << "  --restir-spatial-passes N    Spatial reuse pass count\n"
        << "  --restir-history-length N    Maximum temporal reservoir age\n"
        << "  --restir-max-candidates N    Reservoir M cap (0 is unlimited)\n"
        << "  --restir-normal-threshold N  Temporal/spatial normal threshold\n"
        << "  --restir-depth-threshold N   Temporal/spatial depth threshold\n"
        << "  --restir-bias basic|pairwise Bias correction mode\n"
        << "  --restir-no-temporal         Disable temporal reuse\n"
        << "  --restir-no-spatial          Disable spatial reuse\n"
        << "  --restir-history MODE        reset, continue, or auto\n"
        << "  --save               Save final rendered image\n"
        << "  --save-image PATH    Save benchmark display image to PATH\n"
        << "  --save-linear PATH   Save benchmark linear RGB film as PFM\n";
}

AppOptions parse_options(int argc, char *args[]) {
    AppOptions options;
    std::vector<std::string> positional;

    auto fail = [&](const std::string &message) {
        options.valid = false;
        options.error = message;
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = args[i];
        if (arg == "--bench") {
            options.benchmark.enabled = true;
        } else if (arg == "--save") {
            options.benchmark.save = true;
        } else if (arg == "--save-image" && i + 1 < argc) {
            options.benchmark.image_output_path = args[++i];
        } else if (arg == "--save-linear" && i + 1 < argc) {
            options.benchmark.linear_output_path = args[++i];
        } else if (arg == "--scene-file" && i + 1 < argc) {
            options.scene_file = args[++i];
        } else if (arg == "--integrator" && i + 1 < argc) {
            if (!parse_int_arg(args[++i], options.integrator_id) ||
                options.integrator_id < 0 || options.integrator_id > 7) {
                fail("--integrator expects an integer in the range 0..7.");
                break;
            }
        } else if (arg == "--backend" && i + 1 < argc) {
            const std::string backend = args[++i];
            if (backend == "cpu") {
                options.render.backend = RenderBackend::CPU;
            } else if (backend == "cuda") {
                options.render.backend = RenderBackend::CUDA;
            } else {
                fail("--backend expects 'cpu' or 'cuda'.");
                break;
            }
        } else if (arg == "--mesh-flat") {
            fail("--mesh-flat was removed; MeshAsset is now the default model path.");
            break;
        } else if (arg == "--runs" && i + 1 < argc) {
            if (!parse_int_arg(args[++i], options.benchmark.runs) ||
                options.benchmark.runs <= 0) {
                fail("--runs expects a positive integer.");
                break;
            }
        } else if (arg == "--width" && i + 1 < argc) {
            if (!parse_int_arg(args[++i], options.render.width_override) ||
                options.render.width_override < 2) {
                fail("--width expects an integer of at least 2.");
                break;
            }
        } else if (arg == "--spp" && i + 1 < argc) {
            if (!parse_int_arg(args[++i], options.render.spp_override) ||
                options.render.spp_override <= 0) {
                fail("--spp expects a positive integer.");
                break;
            }
        } else if (arg == "--max-depth" && i + 1 < argc) {
            if (!parse_int_arg(args[++i], options.render.max_depth) ||
                options.render.max_depth <= 0) {
                fail("--max-depth expects a positive integer.");
                break;
            }
        } else if (arg == "--accel" && i + 1 < argc) {
            ++i;
            fail("--accel was removed; LinearBVH is now the default accelerator.");
            break;
        } else if (arg == "--seed" && i + 1 < argc) {
            int parsed_seed = 0;
            if (parse_int_arg(args[++i], parsed_seed) && parsed_seed > 0) {
                options.render.seed = static_cast<unsigned>(parsed_seed);
            } else {
                fail("--seed expects a positive integer.");
                break;
            }
        } else if (arg == "--threads" && i + 1 < argc) {
            if (!parse_int_arg(args[++i], options.render.threads) ||
                options.render.threads < 0) {
                fail("--threads expects a non-negative integer.");
                break;
            }
        } else if (arg == "--sample-clamp" && i + 1 < argc) {
            if (!parse_double_arg(args[++i],
                                  options.render.sample_clamp_override) ||
                options.render.sample_clamp_override < 0.0) {
                fail("--sample-clamp expects a non-negative number.");
                break;
            }
        } else if (arg == "--cuda-batch-size" && i + 1 < argc) {
            int batch_size = 0;
            if (!parse_int_arg(args[++i], batch_size) || batch_size <= 0) {
                fail("--cuda-batch-size expects a positive integer.");
                break;
            }
            options.render.cuda_batch_size =
                static_cast<unsigned>(batch_size);
        } else if (arg == "--restir-light-candidates" && i + 1 < argc) {
            int value = 0;
            if (!parse_int_arg(args[++i], value) || value <= 0) {
                fail("--restir-light-candidates expects a positive integer.");
                break;
            }
            options.render.restir.initial_light_candidates =
                static_cast<std::uint32_t>(value);
        } else if (arg == "--restir-spatial-neighbors" && i + 1 < argc) {
            int value = 0;
            if (!parse_int_arg(args[++i], value) || value <= 0 || value > 64) {
                fail("--restir-spatial-neighbors expects an integer in 1..64.");
                break;
            }
            options.render.restir.spatial_neighbors =
                static_cast<std::uint32_t>(value);
        } else if (arg == "--restir-spatial-passes" && i + 1 < argc) {
            int value = 0;
            if (!parse_int_arg(args[++i], value) || value <= 0) {
                fail("--restir-spatial-passes expects a positive integer.");
                break;
            }
            options.render.restir.spatial_passes =
                static_cast<std::uint32_t>(value);
        } else if (arg == "--restir-history-length" && i + 1 < argc) {
            int value = 0;
            if (!parse_int_arg(args[++i], value) || value <= 0) {
                fail("--restir-history-length expects a positive integer.");
                break;
            }
            options.render.restir.max_history_length =
                static_cast<std::uint32_t>(value);
        } else if (arg == "--restir-max-candidates" && i + 1 < argc) {
            int value = 0;
            if (!parse_int_arg(args[++i], value) || value < 0) {
                fail("--restir-max-candidates expects a non-negative integer.");
                break;
            }
            options.render.restir.max_reservoir_candidates =
                static_cast<std::uint32_t>(value);
        } else if (arg == "--restir-normal-threshold" && i + 1 < argc) {
            double value = 0.0;
            if (!parse_double_arg(args[++i], value) || value < -1.0 ||
                value > 1.0) {
                fail("--restir-normal-threshold expects a number in [-1,1].");
                break;
            }
            options.render.restir.normal_threshold =
                static_cast<float>(value);
        } else if (arg == "--restir-depth-threshold" && i + 1 < argc) {
            double value = 0.0;
            if (!parse_double_arg(args[++i], value) || value < 0.0) {
                fail("--restir-depth-threshold expects a non-negative number.");
                break;
            }
            options.render.restir.depth_threshold =
                static_cast<float>(value);
        } else if (arg == "--restir-bias" && i + 1 < argc) {
            const std::string value = args[++i];
            if (value == "basic") {
                options.render.restir.bias_correction =
                    RestirBiasCorrection::Basic;
            } else if (value == "pairwise") {
                options.render.restir.bias_correction =
                    RestirBiasCorrection::Pairwise;
            } else {
                fail("--restir-bias expects 'basic' or 'pairwise'.");
                break;
            }
        } else if (arg == "--restir-no-temporal") {
            options.render.restir.temporal_reuse = false;
        } else if (arg == "--restir-no-spatial") {
            options.render.restir.spatial_reuse = false;
        } else if (arg == "--restir-history" && i + 1 < argc) {
            const std::string value = args[++i];
            if (value == "reset") {
                options.render.restir.history_mode = RestirHistoryMode::Reset;
            } else if (value == "continue") {
                options.render.restir.history_mode =
                    RestirHistoryMode::Continue;
            } else if (value == "auto") {
                options.render.restir.history_mode = RestirHistoryMode::Auto;
            } else {
                fail("--restir-history expects reset, continue, or auto.");
                break;
            }
        } else if (!arg.empty() && arg[0] == '-') {
            fail("Unknown option '" + arg + "'.");
            break;
        } else {
            positional.push_back(arg);
        }
    }

    if (!options.valid) {
        return options;
    }
    if ((!options.benchmark.image_output_path.empty() ||
         !options.benchmark.linear_output_path.empty()) &&
        !options.benchmark.enabled) {
        fail("--save-image and --save-linear require --bench.");
        return options;
    }
    if (!options.scene_file.empty() && !positional.empty()) {
        fail("--scene-file does not accept positional scene ids; use "
             "--integrator N to select an integrator.");
        return options;
    }
    if (!positional.empty()) {
        if (!parse_int_arg(positional[0].c_str(), options.scene_id)) {
            fail("scene id expects an integer.");
            return options;
        }
    }
    if (positional.size() > 1) {
        if (!parse_int_arg(positional[1].c_str(), options.integrator_id) ||
            options.integrator_id < 0 || options.integrator_id > 7) {
            fail("integrator id expects an integer in the range 0..7.");
            return options;
        }
    }
    if (positional.size() > 2) {
        fail("too many positional arguments.");
        return options;
    }
    return options;
}

#include "app_options.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool parse_int_arg(const char *value, int &out) {
    if (value == nullptr) {
        return false;
    }
    char *end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0') {
        return false;
    }
    out = static_cast<int>(parsed);
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
        << "  --runs N             Benchmark run count\n"
        << "  --width N            Override image width\n"
        << "  --spp N              Override samples per pixel\n"
        << "  --max-depth N        Override integrator max depth\n"
        << "  --seed N             Set render and scene seed (default 1337)\n"
        << "  --threads N          Set render worker count (default hardware)\n"
        << "  --save               Save final rendered image\n";
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
        } else if (arg == "--scene-file" && i + 1 < argc) {
            options.scene_file = args[++i];
        } else if (arg == "--integrator" && i + 1 < argc) {
            if (!parse_int_arg(args[++i], options.integrator_id)) {
                fail("--integrator expects an integer.");
                break;
            }
        } else if (arg == "--mesh-flat") {
            fail("--mesh-flat was removed; FlatMesh is now the default OBJ path.");
            break;
        } else if (arg == "--runs" && i + 1 < argc) {
            if (!parse_int_arg(args[++i], options.benchmark.runs)) {
                fail("--runs expects an integer.");
                break;
            }
        } else if (arg == "--width" && i + 1 < argc) {
            if (!parse_int_arg(args[++i], options.render.width_override)) {
                fail("--width expects an integer.");
                break;
            }
        } else if (arg == "--spp" && i + 1 < argc) {
            if (!parse_int_arg(args[++i], options.render.spp_override)) {
                fail("--spp expects an integer.");
                break;
            }
        } else if (arg == "--max-depth" && i + 1 < argc) {
            if (!parse_int_arg(args[++i], options.render.max_depth)) {
                fail("--max-depth expects an integer.");
                break;
            }
        } else if (arg == "--accel" && i + 1 < argc) {
            ++i;
            fail("--accel was removed; LinearBVH is now the default accelerator.");
            break;
        } else if (arg == "--seed" && i + 1 < argc) {
            int parsed_seed = 0;
            if (parse_int_arg(args[++i], parsed_seed)) {
                options.render.seed = static_cast<unsigned>(parsed_seed);
            } else {
                fail("--seed expects an integer.");
                break;
            }
        } else if (arg == "--threads" && i + 1 < argc) {
            if (!parse_int_arg(args[++i], options.render.threads)) {
                fail("--threads expects an integer.");
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
    if (!options.scene_file.empty() && !positional.empty()) {
        fail("--scene-file does not accept positional scene ids; use "
             "--integrator N to select an integrator.");
        return options;
    }
    if (!positional.empty()) {
        options.scene_id = std::atoi(positional[0].c_str());
    }
    if (positional.size() > 1) {
        options.integrator_id = std::atoi(positional[1].c_str());
    }
    if (options.benchmark.runs < 1) {
        options.benchmark.runs = 1;
    }
    if (options.render.seed == 0) {
        options.render.seed = 1;
    }
    if (options.render.threads < 0) {
        options.render.threads = 0;
    }
    return options;
}

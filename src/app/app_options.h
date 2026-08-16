#ifndef APP_OPTIONS_H
#define APP_OPTIONS_H

#include "render_types.h"

#include <string>

namespace AppDefaults {
constexpr int kMaxDepth = 50;
}

struct RenderOptions {
    int width_override = 0;
    int spp_override = 0;
    int max_depth = AppDefaults::kMaxDepth;
    int threads = 0;
    unsigned seed = 1337;
    double sample_clamp_override = -1.0;
    bool cpu_packed = false;
    bool strict_assets = false;
    RenderBackend backend = RenderBackend::CPU;
    unsigned cuda_batch_size = 0;
    unsigned cuda_samples_per_launch = 0;
    RestirSettings restir;
};

struct BenchmarkOptions {
    bool enabled = false;
    bool save = false;
    int runs = 1;
    std::string image_output_path;
    std::string linear_output_path;
    std::string json_output_path;
};

struct AppOptions {
    int scene_id = 23;
    int integrator_id = 4;
    bool valid = true;
    RenderOptions render;
    BenchmarkOptions benchmark;
    std::string scene_file;
    std::string error;
};

AppOptions parse_options(int argc, char *args[]);
void print_usage();

#endif

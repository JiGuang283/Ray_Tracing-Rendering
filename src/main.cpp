/*The MIT License (MIT)

Copyright (c) 2021-Present, Wencong Yang (yangwc3@mail2.sysu.edu.cn).

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.*/

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <vector>

#include "WindowsApp.h"
#include "direct_light_integrator.h"
#include "mis_path_integrator.h"
#include "path_integrator.h"
#include "pbr_path_integrator.h"
#include "render_buffer.h"
#include "renderer.h"
#include "rr_path_integrator.h"
#include "scenes.h"

namespace RenderConfig {
constexpr int kMaxDepth = 50;
constexpr double kShutterOpen = 0.0;
constexpr double kShutterClose = 1.0;
} // namespace RenderConfig

struct AppOptions {
    int scene_id = 23;
    int integrator_id = 4; // 0: Path, 1: RR, 2: PBR, 3: NEE, 4: MIS
    bool bench = false;
    bool save = false;
    int runs = 1;
    int width_override = 0;
    int spp_override = 0;
    int max_depth = RenderConfig::kMaxDepth;
    std::string accel_mode = "pointer";
    bool mesh_flat = false;
    bool has_seed = false;
    unsigned seed = 0;
};

static bool parse_int_arg(const char *value, int &out) {
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

static AppOptions parse_options(int argc, char *args[]) {
    AppOptions options;
    std::vector<std::string> positional;

    for (int i = 1; i < argc; ++i) {
        std::string arg = args[i];
        if (arg == "--bench") {
            options.bench = true;
        } else if (arg == "--save") {
            options.save = true;
        } else if (arg == "--mesh-flat") {
            options.mesh_flat = true;
        } else if (arg == "--runs" && i + 1 < argc) {
            parse_int_arg(args[++i], options.runs);
        } else if (arg == "--width" && i + 1 < argc) {
            parse_int_arg(args[++i], options.width_override);
        } else if (arg == "--spp" && i + 1 < argc) {
            parse_int_arg(args[++i], options.spp_override);
        } else if (arg == "--max-depth" && i + 1 < argc) {
            parse_int_arg(args[++i], options.max_depth);
        } else if (arg == "--accel" && i + 1 < argc) {
            options.accel_mode = args[++i];
        } else if (arg == "--seed" && i + 1 < argc) {
            int parsed_seed = 0;
            if (parse_int_arg(args[++i], parsed_seed)) {
                options.has_seed = true;
                options.seed = static_cast<unsigned>(parsed_seed);
            }
        } else if (!arg.empty() && arg[0] == '-') {
            std::cerr << "Warning: ignoring unknown option '" << arg << "'."
                      << std::endl;
        } else {
            positional.push_back(arg);
        }
    }

    if (!positional.empty()) {
        options.scene_id = std::atoi(positional[0].c_str());
    }
    if (positional.size() > 1) {
        options.integrator_id = std::atoi(positional[1].c_str());
    }
    if (options.runs < 1) {
        options.runs = 1;
    }
    return options;
}

static shared_ptr<Integrator> make_integrator(int integrator_id) {
    switch (integrator_id) {
    case 0:
        return make_shared<PathIntegrator>();
    case 1:
        return make_shared<RRPathInterator>();
    case 2:
        return make_shared<PBRPathIntegrator>();
    case 3:
        return make_shared<DirectLightIntegrator>();
    case 4:
    default:
        return make_shared<MISPathIntegrator>();
    }
}

static void apply_overrides(SceneConfig &config, const AppOptions &options) {
    if (options.width_override > 0) {
        config.image_width = options.width_override;
    }
    if (options.spp_override > 0) {
        config.samples_per_pixel = options.spp_override;
    }
}

static SceneBuildOptions make_scene_build_options(const AppOptions &options) {
    SceneBuildOptions build_options;
    if (options.accel_mode == "linear") {
        build_options.accel_mode = AccelMode::LinearBVH;
    } else {
        build_options.accel_mode = AccelMode::PointerBVH;
    }
    build_options.use_flat_mesh = options.mesh_flat;
    return build_options;
}

static void apply_seed(const AppOptions &options) {
    if (options.has_seed) {
        set_random_seed(options.seed);
    }
}

static shared_ptr<camera> make_camera(const SceneConfig &config) {
    return make_shared<camera>(
        config.lookfrom, config.lookat, config.vup, config.vfov,
        config.aspect_ratio, config.aperture, config.focus_dist,
        RenderConfig::kShutterOpen, RenderConfig::kShutterClose);
}

static std::string save_rendered_image(const RenderBuffer &render_buffer,
                                       int scene_id, int integrator_id) {
    mkdir("output", 0755);

    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::system_clock::to_time_t(now);
    std::stringstream filename;
    filename << "output/scene" << std::setfill('0') << std::setw(2) << scene_id
             << "_integrator" << integrator_id << "_" << timestamp << ".png";

    std::string output_file = filename.str();
    if (render_buffer.save_to_png(output_file)) {
        std::cout << "Image saved successfully to " << output_file << std::endl;
    } else {
        std::cerr << "Failed to save image to " << output_file << std::endl;
    }
    return output_file;
}

static RenderStats render_once(const AppOptions &options,
                               RenderBuffer *external_buffer = nullptr) {
    apply_seed(options);
    SceneConfig config =
        select_scene(options.scene_id, make_scene_build_options(options));
    apply_overrides(config, options);

    auto cam = make_camera(config);
    int width = config.image_width;
    int height = static_cast<int>(width / config.aspect_ratio);
    auto owned_buffer = make_shared<RenderBuffer>(width, height);
    RenderBuffer &render_buffer =
        external_buffer != nullptr ? *external_buffer : *owned_buffer;

    Renderer renderer;
    renderer.set_samples(config.samples_per_pixel);
    renderer.set_integrator(make_integrator(options.integrator_id));
    renderer.set_max_depth(options.max_depth);

    return renderer.render(config.world, cam, config.background, render_buffer,
                           config.lights);
}

static int run_benchmark(const AppOptions &options) {
    std::vector<double> seconds;
    seconds.reserve(options.runs);
    RenderStats last_stats;
    shared_ptr<RenderBuffer> saved_buffer;

    for (int run = 1; run <= options.runs; ++run) {
        apply_seed(options);
        SceneConfig config =
            select_scene(options.scene_id, make_scene_build_options(options));
        apply_overrides(config, options);
        int width = config.image_width;
        int height = static_cast<int>(width / config.aspect_ratio);
        auto render_buffer = make_shared<RenderBuffer>(width, height);

        auto cam = make_camera(config);
        Renderer renderer;
        renderer.set_samples(config.samples_per_pixel);
        renderer.set_integrator(make_integrator(options.integrator_id));
        renderer.set_max_depth(options.max_depth);

        last_stats = renderer.render(config.world, cam, config.background,
                                     *render_buffer, config.lights);
        seconds.push_back(last_stats.seconds);
        saved_buffer = render_buffer;

        double samples_per_second =
            last_stats.seconds > 0.0
                ? last_stats.sample_count / last_stats.seconds
                : 0.0;
        std::cout << "BENCH_RUN"
                  << " run=" << run << " scene=" << options.scene_id
                  << " integrator=" << options.integrator_id
                  << " width=" << last_stats.width
                  << " height=" << last_stats.height
                  << " spp=" << last_stats.samples_per_pixel
                  << " samples=" << last_stats.sample_count
                  << " seconds=" << last_stats.seconds
                  << " samples_per_second=" << samples_per_second
                  << " accel=" << options.accel_mode
                  << " mesh_flat=" << (options.mesh_flat ? 1 : 0)
                  << std::endl;
    }

    auto sorted = seconds;
    std::sort(sorted.begin(), sorted.end());
    double median = sorted[sorted.size() / 2];
    double samples_per_second =
        median > 0.0 ? last_stats.sample_count / median : 0.0;
    std::cout << "BENCH_SUMMARY"
              << " runs=" << options.runs << " scene=" << options.scene_id
              << " integrator=" << options.integrator_id
              << " width=" << last_stats.width
              << " height=" << last_stats.height
              << " spp=" << last_stats.samples_per_pixel
              << " median_seconds=" << median
              << " median_samples_per_second=" << samples_per_second
              << " accel=" << options.accel_mode
              << " mesh_flat=" << (options.mesh_flat ? 1 : 0) << std::endl;

    if (options.save && saved_buffer) {
        save_rendered_image(*saved_buffer, options.scene_id,
                            options.integrator_id);
    }

    return 0;
}

static int run_windowed(const AppOptions &options) {
    apply_seed(options);
    SceneConfig config =
        select_scene(options.scene_id, make_scene_build_options(options));
    apply_overrides(config, options);

    auto cam = make_camera(config);
    int width = config.image_width;
    int height = static_cast<int>(width / config.aspect_ratio);
    auto render_buffer = make_shared<RenderBuffer>(width, height);

    Renderer renderer;
    renderer.set_samples(config.samples_per_pixel);
    renderer.set_integrator(make_integrator(options.integrator_id));
    renderer.set_max_depth(options.max_depth);

    WindowsApp::ptr winApp =
        WindowsApp::getInstance(width, height, "CGAssignment4: Ray Tracing");
    if (winApp == nullptr) {
        std::cerr << "Error: failed to create a window handler" << std::endl;
        return -1;
    }

    std::thread renderingThread([&renderer, world = config.world, cam,
                                 render_buffer, bg = config.background,
                                 lights = config.lights]() {
        renderer.render(world, cam, bg, *render_buffer, lights);
    });

    while (!winApp->shouldWindowClose()) {
        winApp->processEvent();
        winApp->updateScreenSurface(render_buffer->get_data(), width, height);
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
    }

    renderer.cancel();
    if (renderingThread.joinable()) {
        renderingThread.join();
    }

    std::cout << "Saving rendered image..." << std::endl;
    save_rendered_image(*render_buffer, options.scene_id, options.integrator_id);
    return 0;
}

int main(int argc, char *args[]) {
    AppOptions options = parse_options(argc, args);
    if (options.bench) {
        return run_benchmark(options);
    }
    return run_windowed(options);
}

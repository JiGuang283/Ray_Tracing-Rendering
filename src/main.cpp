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
#include <exception>
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
#include "scene_loader.h"
#include "scenes.h"

namespace RenderConfig {
constexpr int kMaxDepth = 50;
constexpr double kShutterOpen = 0.0;
constexpr double kShutterClose = 1.0;
} // namespace RenderConfig

struct RenderOptions {
    int width_override = 0;
    int spp_override = 0;
    int max_depth = RenderConfig::kMaxDepth;
    int threads = 0;
    unsigned seed = 1337;
};

struct BenchmarkOptions {
    bool enabled = false;
    bool save = false;
    int runs = 1;
};

struct AppOptions {
    int scene_id = 23;
    int integrator_id = 4; // 0: Path, 1: RR, 2: PBR, 3: NEE, 4: MIS
    bool valid = true;
    RenderOptions render;
    BenchmarkOptions benchmark;
    std::string scene_file;
    std::string error;
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

static void print_usage() {
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

static AppOptions parse_options(int argc, char *args[]) {
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
    if (options.render.width_override > 0) {
        config.preset.image_width = options.render.width_override;
    }
    if (options.render.spp_override > 0) {
        config.preset.samples_per_pixel = options.render.spp_override;
    }
}

static void apply_seed(const AppOptions &options) {
    set_random_seed(options.render.seed);
}

static SceneConfig load_scene_config(const AppOptions &options) {
    if (!options.scene_file.empty()) {
        return load_scene_file(options.scene_file);
    }
    return select_scene(options.scene_id);
}

static void configure_renderer(Renderer &renderer, const SceneConfig &config,
                               const AppOptions &options) {
    renderer.set_samples(config.preset.samples_per_pixel);
    renderer.set_seed(options.render.seed);
    renderer.set_thread_count(options.render.threads);
    renderer.set_integrator(make_integrator(options.integrator_id));
    renderer.set_max_depth(options.render.max_depth);
}

static shared_ptr<camera> make_camera(const SceneConfig &config) {
    return make_shared<camera>(
        config.camera.lookfrom, config.camera.lookat, config.camera.vup, config.camera.vfov,
        config.camera.aspect_ratio, config.camera.aperture, config.camera.focus_dist,
        RenderConfig::kShutterOpen, RenderConfig::kShutterClose);
}

static std::string save_rendered_image(const RenderBuffer &render_buffer,
                                       int scene_id, int integrator_id) {
    mkdir("output", 0755);

    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
                         now.time_since_epoch())
                         .count();
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
    SceneConfig config = load_scene_config(options);
    apply_overrides(config, options);

    auto cam = make_camera(config);
    int width = config.preset.image_width;
    int height = static_cast<int>(width / config.camera.aspect_ratio);
    auto owned_buffer = make_shared<RenderBuffer>(width, height);
    RenderBuffer &render_buffer =
        external_buffer != nullptr ? *external_buffer : *owned_buffer;

    Renderer renderer;
    configure_renderer(renderer, config, options);

    return renderer.render(config.scene.world, cam, config.preset.background, render_buffer,
                           config.scene.lights);
}

static int run_benchmark(const AppOptions &options) {
    std::vector<double> seconds;
    seconds.reserve(options.benchmark.runs);
    RenderStats last_stats;
    shared_ptr<RenderBuffer> saved_buffer;

    for (int run = 1; run <= options.benchmark.runs; ++run) {
        apply_seed(options);
        SceneConfig config = load_scene_config(options);
        apply_overrides(config, options);
        int width = config.preset.image_width;
        int height = static_cast<int>(width / config.camera.aspect_ratio);
        auto render_buffer = make_shared<RenderBuffer>(width, height);

        auto cam = make_camera(config);
        Renderer renderer;
        configure_renderer(renderer, config, options);

        last_stats = renderer.render(config.scene.world, cam, config.preset.background,
                                     *render_buffer, config.scene.lights);
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
                  << " seed=" << last_stats.seed
                  << " threads=" << last_stats.threads
                  << std::endl;
    }

    auto sorted = seconds;
    std::sort(sorted.begin(), sorted.end());
    double median = sorted[sorted.size() / 2];
    double samples_per_second =
        median > 0.0 ? last_stats.sample_count / median : 0.0;
    std::cout << "BENCH_SUMMARY"
              << " runs=" << options.benchmark.runs << " scene=" << options.scene_id
              << " integrator=" << options.integrator_id
              << " width=" << last_stats.width
              << " height=" << last_stats.height
              << " spp=" << last_stats.samples_per_pixel
              << " median_seconds=" << median
              << " median_samples_per_second=" << samples_per_second
              << " seed=" << last_stats.seed
              << " threads=" << last_stats.threads << std::endl;

    if (options.benchmark.save && saved_buffer) {
        save_rendered_image(*saved_buffer, options.scene_id,
                            options.integrator_id);
    }

    return 0;
}

static int run_windowed(const AppOptions &options) {
    apply_seed(options);
    SceneConfig config = load_scene_config(options);
    apply_overrides(config, options);

    auto cam = make_camera(config);
    int width = config.preset.image_width;
    int height = static_cast<int>(width / config.camera.aspect_ratio);
    auto render_buffer = make_shared<RenderBuffer>(width, height);

    Renderer renderer;
    configure_renderer(renderer, config, options);

    WindowsApp::ptr winApp =
        WindowsApp::getInstance(width, height, "CGAssignment4: Ray Tracing");
    if (winApp == nullptr) {
        std::cerr << "Error: failed to create a window handler" << std::endl;
        return -1;
    }

    std::thread renderingThread([&renderer, world = config.scene.world, cam,
                                 render_buffer, bg = config.preset.background,
                                 lights = config.scene.lights]() {
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
    if (!options.valid) {
        std::cerr << "Error: " << options.error << std::endl;
        print_usage();
        return 1;
    }
    try {
        if (options.benchmark.enabled) {
            return run_benchmark(options);
        }
        return run_windowed(options);
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return 1;
    }
}

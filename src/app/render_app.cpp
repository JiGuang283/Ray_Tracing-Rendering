#include "render_app.h"

#include "WindowsApp.h"
#include "direct_light_integrator.h"
#include "image_output.h"
#include "mis_path_integrator.h"
#include "path_integrator.h"
#include "pbr_path_integrator.h"
#include "render_buffer.h"
#include "renderer.h"
#include "rr_path_integrator.h"
#include "scene_loader.h"
#include "scenes.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

namespace {

constexpr double kShutterOpen = 0.0;
constexpr double kShutterClose = 1.0;

shared_ptr<Integrator> make_integrator(int integrator_id) {
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

void apply_overrides(SceneConfig &config, const AppOptions &options) {
    if (options.render.width_override > 0) {
        config.preset.image_width = options.render.width_override;
    }
    if (options.render.spp_override > 0) {
        config.preset.samples_per_pixel = options.render.spp_override;
    }
    if (options.render.sample_clamp_override >= 0.0) {
        config.preset.sample_clamp = options.render.sample_clamp_override;
    }
}

void apply_seed(const AppOptions &options) {
    set_random_seed(options.render.seed);
}

SceneConfig load_scene_config(const AppOptions &options) {
    if (!options.scene_file.empty()) {
        return load_scene_file(options.scene_file);
    }
    return select_scene(options.scene_id);
}

void configure_renderer(Renderer &renderer, const SceneConfig &config,
                        const AppOptions &options) {
    renderer.set_samples(config.preset.samples_per_pixel);
    renderer.set_seed(options.render.seed);
    renderer.set_thread_count(options.render.threads);
    renderer.set_sample_clamp(config.preset.sample_clamp);
    renderer.set_color_pipeline(config.preset.color_pipeline);
    renderer.set_integrator(make_integrator(options.integrator_id));
    renderer.set_max_depth(options.render.max_depth);
}

shared_ptr<camera> make_camera(const SceneConfig &config) {
    return make_shared<camera>(
        config.camera.lookfrom, config.camera.lookat, config.camera.vup,
        config.camera.vfov, config.camera.aspect_ratio, config.camera.aperture,
        config.camera.focus_dist, kShutterOpen, kShutterClose);
}

} // namespace

int run_benchmark(const AppOptions &options) {
    std::vector<double> seconds;
    seconds.reserve(options.benchmark.runs);
    RenderStats last_stats;
    long long clamped_samples = 0;
    long long invalid_samples = 0;
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

        last_stats = renderer.render(config.scene.world, cam,
                                     config.preset.background, *render_buffer,
                                     config.scene.lights);
        seconds.push_back(last_stats.seconds);
        clamped_samples += last_stats.clamped_samples;
        invalid_samples += last_stats.invalid_samples;
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
                  << " clamped_samples=" << last_stats.clamped_samples
                  << " invalid_samples=" << last_stats.invalid_samples
                  << std::endl;
    }

    auto sorted = seconds;
    std::sort(sorted.begin(), sorted.end());
    double median = sorted[sorted.size() / 2];
    double samples_per_second =
        median > 0.0 ? last_stats.sample_count / median : 0.0;
    std::cout << "BENCH_SUMMARY"
              << " runs=" << options.benchmark.runs
              << " scene=" << options.scene_id
              << " integrator=" << options.integrator_id
              << " width=" << last_stats.width
              << " height=" << last_stats.height
              << " spp=" << last_stats.samples_per_pixel
              << " median_seconds=" << median
              << " median_samples_per_second=" << samples_per_second
              << " seed=" << last_stats.seed
              << " threads=" << last_stats.threads
              << " clamped_samples=" << clamped_samples
              << " invalid_samples=" << invalid_samples << std::endl;

    if (options.benchmark.save && saved_buffer) {
        save_rendered_image(*saved_buffer, options.scene_id,
                            options.integrator_id);
    }

    return 0;
}

int run_windowed(const AppOptions &options) {
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

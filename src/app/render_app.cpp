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
#include "scene_ir.h"
#include "scenes.h"

#if RAYTRACER_HAS_CUDA
#include "cuda/cuda_renderer.h"
#include "render_data/scene_compiler.h"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
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

void apply_overrides(SceneIR &ir, const AppOptions &options) {
    if (options.render.width_override > 0) {
        ir.preset.image_width = options.render.width_override;
    }
    if (options.render.spp_override > 0) {
        ir.preset.samples_per_pixel = options.render.spp_override;
    }
    if (options.render.sample_clamp_override >= 0.0) {
        ir.preset.sample_clamp = options.render.sample_clamp_override;
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

std::string scene_source_path(const AppOptions &options) {
    return options.scene_file.empty() ? scene_path(options.scene_id)
                                      : options.scene_file;
}

SceneIR load_scene_ir_config(const AppOptions &options) {
    return load_scene_ir_file(scene_source_path(options));
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

std::string log_token(std::string value) {
    for (char &character : value) {
        if (character == ' ' || character == '\t') {
            character = '_';
        }
    }
    return value;
}

#if RAYTRACER_HAS_CUDA
cuda_backend::CudaRendererSettings make_cuda_settings(
    const SceneIR &ir, const AppOptions &options) {
    if (options.render.max_depth <= 0) {
        throw std::invalid_argument("CUDA max depth must be positive");
    }
    cuda_backend::CudaRendererSettings settings;
    settings.integrator_id = options.integrator_id;
    settings.max_depth =
        static_cast<std::uint32_t>(options.render.max_depth);
    settings.samples_per_pixel =
        static_cast<std::uint32_t>(ir.preset.samples_per_pixel);
    settings.seed = options.render.seed;
    settings.batch_size = options.render.cuda_batch_size;
    settings.sample_clamp = static_cast<float>(ir.preset.sample_clamp);
    settings.color_pipeline = ir.preset.color_pipeline;
    return settings;
}
#else
[[noreturn]] void throw_cuda_not_built() {
    throw std::runtime_error(
        "CUDA backend was not built. Configure with "
        "-DRAYTRACER_ENABLE_CUDA=ON and use the CUDA build directory.");
}
#endif

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
        shared_ptr<RenderBuffer> render_buffer;
        if (options.render.backend == RenderBackend::CPU) {
            SceneConfig config = load_scene_config(options);
            apply_overrides(config, options);
            const int width = config.preset.image_width;
            const int height =
                static_cast<int>(width / config.camera.aspect_ratio);
            render_buffer = make_shared<RenderBuffer>(width, height);

            auto cam = make_camera(config);
            Renderer renderer;
            configure_renderer(renderer, config, options);
            last_stats = renderer.render(
                config.scene.world, cam, config.preset.background,
                *render_buffer, config.scene.lights);
        } else {
#if RAYTRACER_HAS_CUDA
            SceneIR ir = load_scene_ir_config(options);
            apply_overrides(ir, options);
            const int width = ir.preset.image_width;
            const int height =
                static_cast<int>(width / ir.camera.aspect_ratio);
            render_buffer = make_shared<RenderBuffer>(width, height);
            const CompiledScene scene = compile_scene(ir);
            last_stats = cuda_backend::render_cuda(
                scene, make_cuda_settings(ir, options), *render_buffer);
#else
            throw_cuda_not_built();
#endif
        }
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
                  << " backend=" << last_stats.backend
                  << " integrator=" << options.integrator_id
                  << " width=" << last_stats.width
                  << " height=" << last_stats.height
                  << " spp=" << last_stats.samples_per_pixel
                  << " samples=" << last_stats.sample_count
                  << " seconds=" << last_stats.seconds
                  << " samples_per_second=" << samples_per_second
                  << " seed=" << last_stats.seed
                  << " threads=" << last_stats.threads
                  << " device=" << log_token(last_stats.device_name)
                  << " upload_seconds=" << last_stats.upload_seconds
                  << " device_seconds=" << last_stats.device_seconds
                  << " scene_bytes=" << last_stats.scene_bytes
                  << " workspace_bytes=" << last_stats.workspace_bytes
                  << " batch_size=" << last_stats.batch_size
                  << " batch_count=" << last_stats.batch_count
                  << " traversal_steps=" << last_stats.traversal_steps
                  << " shadow_rays=" << last_stats.shadow_rays
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
              << " backend=" << last_stats.backend
              << " integrator=" << options.integrator_id
              << " width=" << last_stats.width
              << " height=" << last_stats.height
              << " spp=" << last_stats.samples_per_pixel
              << " median_seconds=" << median
              << " median_samples_per_second=" << samples_per_second
              << " seed=" << last_stats.seed
              << " threads=" << last_stats.threads
              << " device=" << log_token(last_stats.device_name)
              << " upload_seconds=" << last_stats.upload_seconds
              << " device_seconds=" << last_stats.device_seconds
              << " scene_bytes=" << last_stats.scene_bytes
              << " workspace_bytes=" << last_stats.workspace_bytes
              << " batch_size=" << last_stats.batch_size
              << " batch_count=" << last_stats.batch_count
              << " clamped_samples=" << clamped_samples
              << " invalid_samples=" << invalid_samples << std::endl;

    if (options.benchmark.save && saved_buffer) {
        save_rendered_image(*saved_buffer, options.scene_id,
                            options.integrator_id);
    }

    return 0;
}

int run_windowed(const AppOptions &options) {
    if (options.render.backend == RenderBackend::CUDA) {
#if RAYTRACER_HAS_CUDA
        apply_seed(options);
        SceneIR ir = load_scene_ir_config(options);
        apply_overrides(ir, options);
        const int width = ir.preset.image_width;
        const int height = static_cast<int>(width / ir.camera.aspect_ratio);
        auto render_buffer = make_shared<RenderBuffer>(width, height);
        CompiledScene scene = compile_scene(ir);

        WindowsApp::ptr win_app = WindowsApp::getInstance(
            width, height, "CGAssignment4: CUDA Ray Tracing");
        if (win_app == nullptr) {
            std::cerr << "Error: failed to create a window handler"
                      << std::endl;
            return -1;
        }
        std::exception_ptr render_error;
        std::atomic<bool> render_failed{false};
        std::atomic<bool> cancel_render{false};
        std::thread rendering_thread([&]() {
            try {
                cuda_backend::render_cuda(
                    scene, make_cuda_settings(ir, options), *render_buffer,
                    &cancel_render);
            } catch (...) {
                render_error = std::current_exception();
                render_failed.store(true);
            }
        });
        while (!win_app->shouldWindowClose() && !render_failed.load()) {
            win_app->processEvent();
            win_app->updateScreenSurface(render_buffer->get_data(), width,
                                         height);
            std::this_thread::sleep_for(std::chrono::milliseconds(33));
        }
        cancel_render.store(true);
        if (rendering_thread.joinable()) {
            rendering_thread.join();
        }
        if (render_error) {
            std::rethrow_exception(render_error);
        }
        std::cout << "Saving rendered image..." << std::endl;
        save_rendered_image(*render_buffer, options.scene_id,
                            options.integrator_id);
        return 0;
#else
        throw_cuda_not_built();
#endif
    }

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

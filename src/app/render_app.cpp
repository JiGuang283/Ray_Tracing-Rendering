#include "render_app.h"

#include "raytracer/build_config.h"

#include "WindowsApp.h"
#include "direct_light_integrator.h"
#include "image_output.h"
#include "mis_path_integrator.h"
#include "path_integrator.h"
#include "pbr_path_integrator.h"
#include "preview_surface.h"
#include "render_buffer.h"
#include "render_result.h"
#include "render_types.h"
#include "renderer.h"
#include "rr_path_integrator.h"
#include "scene_loader.h"
#include "scene_ir.h"
#include "scenes.h"

#if RAYTRACER_HAS_CUDA
#include "cuda_renderer.h"
#include "scene_compiler.h"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <functional>
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
    (void)config;
    renderer.set_integrator(make_integrator(options.integrator_id));
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

#if !RAYTRACER_HAS_CUDA
[[noreturn]] void throw_cuda_not_built() {
    throw std::runtime_error(
        "CUDA backend was not built. Configure with "
        "-DRAYTRACER_ENABLE_CUDA=ON and use the CUDA build directory.");
}
#endif

RenderRequest make_render_request(const CameraConfig &camera,
                                  const RenderPreset &preset,
                                  const AppOptions &options) {
    RenderRequest request;
    request.extent = make_image_extent(preset.image_width, camera.aspect_ratio);
    request.integrator = integrator_kind_from_id(options.integrator_id);
    request.samples_per_pixel =
        static_cast<std::uint32_t>(preset.samples_per_pixel);
    request.max_depth =
        static_cast<std::uint32_t>(options.render.max_depth);
    request.seed = options.render.seed;
    request.threads = static_cast<std::uint32_t>(options.render.threads);
    request.cuda_batch_size = options.render.cuda_batch_size;
    request.sample_clamp = preset.sample_clamp;
    request.color_pipeline = preset.color_pipeline;
    validate_render_request(request);
    return request;
}

using WindowRenderFunction = std::function<RenderResult(
    const CancellationToken &, PreviewSurface *)>;

int run_window_render(const AppOptions &options, const RenderRequest &request,
                      const std::string &title,
                      const WindowRenderFunction &render) {
    const int width = static_cast<int>(request.extent.width);
    const int height = static_cast<int>(request.extent.height);
    WindowsApp::ptr window = WindowsApp::getInstance(width, height, title);
    if (window == nullptr) {
        std::cerr << "Error: failed to create a window handler" << std::endl;
        return -1;
    }

    PreviewSurface preview(request.extent);
    CancellationSource cancellation;
    RenderResult result;
    std::exception_ptr render_error;
    std::atomic<bool> render_failed{false};
    std::thread rendering_thread([&]() {
        try {
            result = render(cancellation.token(), &preview);
        } catch (...) {
            render_error = std::current_exception();
            render_failed.store(true, std::memory_order_release);
        }
    });

    while (!window->shouldWindowClose() &&
           !render_failed.load(std::memory_order_acquire)) {
        window->processEvent();
        const RenderBuffer snapshot = preview.snapshot();
        window->updateScreenSurface(snapshot.get_data(), width, height);
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
    }

    cancellation.cancel();
    if (rendering_thread.joinable()) {
        rendering_thread.join();
    }
    if (render_error) {
        std::rethrow_exception(render_error);
    }

    std::cout << "Saving rendered image..." << std::endl;
    save_rendered_image(result.display, options.scene_id,
                        options.integrator_id);
    return 0;
}

} // namespace

int run_benchmark(const AppOptions &options) {
    std::vector<double> seconds;
    seconds.reserve(options.benchmark.runs);
    std::vector<double> upload_seconds;
    upload_seconds.reserve(options.benchmark.runs);
    std::vector<double> device_seconds;
    device_seconds.reserve(options.benchmark.runs);
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
            const RenderRequest request =
                make_render_request(config.camera, config.preset, options);
            auto cam = make_camera(config);
            Renderer renderer;
            configure_renderer(renderer, config, options);
            RenderResult result = renderer.render(
                config.scene.world, cam, config.preset.background,
                config.scene.lights, request);
            last_stats = result.stats;
            render_buffer =
                make_shared<RenderBuffer>(std::move(result.display));
        } else {
#if RAYTRACER_HAS_CUDA
            SceneIR ir = load_scene_ir_config(options);
            apply_overrides(ir, options);
            const RenderRequest request =
                make_render_request(ir.camera, ir.preset, options);
            const CompiledScene scene = compile_scene(ir);
            RenderResult result = cuda_backend::render_cuda(scene, request);
            last_stats = result.stats;
            render_buffer =
                make_shared<RenderBuffer>(std::move(result.display));
#else
            throw_cuda_not_built();
#endif
        }
        seconds.push_back(last_stats.seconds);
        upload_seconds.push_back(last_stats.upload_seconds);
        device_seconds.push_back(last_stats.device_seconds);
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
                  << " requested_samples=" << last_stats.requested_samples
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
                  << " cancelled=" << (last_stats.cancelled ? 1 : 0)
                  << std::endl;
    }

    auto sorted = seconds;
    std::sort(sorted.begin(), sorted.end());
    double median = sorted[sorted.size() / 2];
    std::sort(upload_seconds.begin(), upload_seconds.end());
    std::sort(device_seconds.begin(), device_seconds.end());
    const double median_upload_seconds =
        upload_seconds[upload_seconds.size() / 2];
    const double median_device_seconds =
        device_seconds[device_seconds.size() / 2];
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
              << " median_upload_seconds=" << median_upload_seconds
              << " median_device_seconds=" << median_device_seconds
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
        const RenderRequest request =
            make_render_request(ir.camera, ir.preset, options);
        auto scene = std::make_shared<CompiledScene>(compile_scene(ir));
        return run_window_render(
            options, request, "CGAssignment4: CUDA Ray Tracing",
            [scene, request](const CancellationToken &cancel,
                             PreviewSurface *preview) {
                return cuda_backend::render_cuda(*scene, request, cancel,
                                                 preview);
            });
#else
        throw_cuda_not_built();
#endif
    }

    apply_seed(options);
    SceneConfig config = load_scene_config(options);
    apply_overrides(config, options);

    const RenderRequest request =
        make_render_request(config.camera, config.preset, options);
    auto cam = make_camera(config);
    auto renderer = std::make_shared<Renderer>();
    configure_renderer(*renderer, config, options);
    return run_window_render(
        options, request, "CGAssignment4: Ray Tracing",
        [renderer, world = config.scene.world, cam,
         background = config.preset.background, lights = config.scene.lights,
         request](const CancellationToken &cancel, PreviewSurface *preview) {
            return renderer->render(world, cam, background, lights, request,
                                    cancel, preview);
        });
}

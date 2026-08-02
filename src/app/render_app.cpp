#include "render_app.h"

#include "raytracer/build_config.h"

#include "WindowsApp.h"
#include "cpu_render_session.h"
#include "image_output.h"
#include "preview_surface.h"
#include "render_buffer.h"
#include "render_result.h"
#include "render_session.h"
#include "render_types.h"
#include "rtweekend.h"
#include "scene_ir.h"
#include "scenes.h"

#if RAYTRACER_HAS_CUDA
#include "cuda_renderer.h"
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

std::string scene_source_path(const AppOptions &options) {
    return options.scene_file.empty() ? scene_path(options.scene_id)
                                      : options.scene_file;
}

SceneIR load_scene_ir_config(const AppOptions &options) {
    return load_scene_ir_file(scene_source_path(options));
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

std::shared_ptr<IRenderSession>
make_render_session(const AppOptions &options, const SceneIR &ir) {
    if (options.render.backend == RenderBackend::CPU) {
        return std::shared_ptr<IRenderSession>(make_cpu_render_session(ir));
    }
#if RAYTRACER_HAS_CUDA
    return std::shared_ptr<IRenderSession>(
        cuda_backend::make_cuda_render_session(ir));
#else
    throw_cuda_not_built();
#endif
}

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

int run_window_render(const AppOptions &options, const RenderRequest &request,
                      const std::string &title,
                      const std::shared_ptr<IRenderSession> &session) {
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
            result = session->render(request, cancellation.token(), &preview);
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
    apply_seed(options);
    SceneIR ir = load_scene_ir_config(options);
    apply_overrides(ir, options);
    const RenderRequest request =
        make_render_request(ir.camera, ir.preset, options);
    std::shared_ptr<IRenderSession> session =
        make_render_session(options, ir);
    const PreparationStats preparation = session->preparation_stats();

    std::vector<double> seconds;
    seconds.reserve(options.benchmark.runs);
    std::vector<double> device_seconds;
    device_seconds.reserve(options.benchmark.runs);
    RenderStats last_stats;
    long long clamped_samples = 0;
    long long invalid_samples = 0;
    RenderBuffer saved_buffer;
    bool has_saved_buffer = false;

    std::cout << "BENCH_PREP"
              << " scene=" << options.scene_id
              << " backend="
              << (options.render.backend == RenderBackend::CPU ? "cpu"
                                                                : "cuda")
              << " compile_seconds=" << preparation.compile_seconds
              << " upload_seconds=" << preparation.upload_seconds
              << " scene_bytes=" << preparation.scene_bytes << std::endl;

    for (int run = 1; run <= options.benchmark.runs; ++run) {
        RenderResult result = session->render(request, {});
        last_stats = result.stats;
        seconds.push_back(last_stats.seconds);
        device_seconds.push_back(last_stats.device_seconds);
        clamped_samples += last_stats.clamped_samples;
        invalid_samples += last_stats.invalid_samples;
        saved_buffer = std::move(result.display);
        has_saved_buffer = true;

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
                  << " compile_seconds=" << last_stats.compile_seconds
                  << " upload_seconds=" << last_stats.upload_seconds
                  << " device_seconds=" << last_stats.device_seconds
                  << " scene_bytes=" << last_stats.scene_bytes
                  << " workspace_bytes=" << last_stats.workspace_bytes
                  << " workspace_generation="
                  << last_stats.workspace_generation
                  << " workspace_pixel_capacity="
                  << last_stats.workspace_pixel_capacity
                  << " workspace_path_capacity="
                  << last_stats.workspace_path_capacity
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
    std::sort(device_seconds.begin(), device_seconds.end());
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
              << " compile_seconds=" << preparation.compile_seconds
              << " upload_seconds=" << preparation.upload_seconds
              << " median_device_seconds=" << median_device_seconds
              << " scene_bytes=" << last_stats.scene_bytes
              << " workspace_bytes=" << last_stats.workspace_bytes
              << " workspace_generation="
              << last_stats.workspace_generation
              << " workspace_pixel_capacity="
              << last_stats.workspace_pixel_capacity
              << " workspace_path_capacity="
              << last_stats.workspace_path_capacity
              << " batch_size=" << last_stats.batch_size
              << " batch_count=" << last_stats.batch_count
              << " clamped_samples=" << clamped_samples
              << " invalid_samples=" << invalid_samples << std::endl;

    if (options.benchmark.save && has_saved_buffer) {
        save_rendered_image(saved_buffer, options.scene_id,
                            options.integrator_id);
    }

    return 0;
}

int run_windowed(const AppOptions &options) {
    apply_seed(options);
    SceneIR ir = load_scene_ir_config(options);
    apply_overrides(ir, options);
    const RenderRequest request =
        make_render_request(ir.camera, ir.preset, options);
    std::shared_ptr<IRenderSession> session =
        make_render_session(options, ir);
    const std::string title =
        options.render.backend == RenderBackend::CUDA
            ? "CGAssignment4: CUDA Ray Tracing"
            : "CGAssignment4: Ray Tracing";
    return run_window_render(
        options, request, title, session);
}

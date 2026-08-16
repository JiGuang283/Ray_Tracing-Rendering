#include "cpu_packed_render_session.h"

#include "beauty_film.h"
#include "film.h"
#include "packed_transport.h"
#include "packed_transport_core.h"
#include "scene_compiler.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

double luminance(const Float3 &value) {
    return 0.2126 * value.x + 0.7152 * value.y + 0.0722 * value.z;
}

bool finite(const Float3 &value) {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z);
}

class CpuPackedRenderSession final : public IRenderSession {
  public:
    explicit CpuPackedRenderSession(const SceneIR &ir) {
        const auto begin = std::chrono::steady_clock::now();
        m_scene = compile_scene(ir);
        m_time0 = ir.time0;
        m_time1 = ir.time1;
        m_default_camera = ir.camera;
        const auto end = std::chrono::steady_clock::now();
        m_preparation.compile_seconds =
            std::chrono::duration<double>(end - begin).count();
        m_preparation.scene_bytes =
            compiled_scene_stats(m_scene).bytes;
    }

    const PreparationStats &preparation_stats() const noexcept override {
        return m_preparation;
    }

    RenderResult render(const RenderRequest &request,
                        const CancellationToken &cancel,
                        PreviewSurface *preview) override {
        reset_history();
        return render_with_camera(request, m_default_camera, cancel,
                                  preview);
    }

    RenderResult render_frame(const RenderFrameRequest &request,
                              const CancellationToken &cancel,
                              PreviewSurface *preview) override {
        validate_render_frame_request(request);
        return render_with_camera(request.render, request.camera, cancel,
                                  preview);
    }

    void reset_history() override {
    }

  private:
    RenderResult render_with_camera(const RenderRequest &request,
                                    const CameraConfig &camera_config,
                                    const CancellationToken &cancel,
                                    PreviewSurface *preview) {
        validate_render_request(request);
        const IntegratorDescriptor &descriptor =
            integrator_descriptor(request.integrator);
        if (descriptor.execution_model !=
                IntegratorExecutionModel::WavefrontPath ||
            !descriptor.supports_cpu) {
            throw std::invalid_argument(
                "CPU packed backend supports integrators 0 through 4");
        }
        if (m_rendering.exchange(true)) {
            throw std::logic_error(
                "CPU packed render session is already rendering");
        }
        struct Guard {
            std::atomic<bool> &state;
            ~Guard() {
                state.store(false, std::memory_order_relaxed);
            }
        } guard{m_rendering};

        const auto begin = std::chrono::steady_clock::now();
        CompiledSceneView view = make_scene_view(m_scene);
        view.camera = compile_packed_camera(camera_config, m_time0, m_time1);

        PackedTransportSettings settings;
        settings.policy = integrator_policy(request.integrator);
        settings.max_depth = request.max_depth;

        Film film(request.extent, request.color_pipeline, preview);

        const int width = static_cast<int>(request.extent.width);
        const int height = static_cast<int>(request.extent.height);
        constexpr int kTileSize = 32;
        const int tiles_x = (width + kTileSize - 1) / kTileSize;
        const int tiles_y = (height + kTileSize - 1) / kTileSize;
        const int total_tiles = tiles_x * tiles_y;
        std::atomic<int> next_tile(0);

        const int thread_count =
            request.threads > 0
                ? static_cast<int>(request.threads)
                : static_cast<int>(
                      std::max(1u, std::thread::hardware_concurrency()));

        struct WorkerStats {
            std::uint64_t samples = 0;
            std::uint64_t invalid = 0;
            std::uint64_t clamped = 0;
            std::uint64_t traversal_steps = 0;
            std::uint64_t shadow_rays = 0;
        };
        std::vector<WorkerStats> worker_stats(
            static_cast<std::size_t>(thread_count));
        std::atomic<bool> failed{false};
        std::mutex exception_mutex;
        std::exception_ptr first_exception;

        auto worker = [&](int worker_id) {
            try {
                WorkerStats &stats = worker_stats[worker_id];
                while (!failed.load(std::memory_order_relaxed) &&
                       !cancel.is_cancelled()) {
                    const int tile = next_tile.fetch_add(1);
                    if (tile >= total_tiles) {
                        break;
                    }
                    const int tile_y = (tiles_y - 1) - tile / tiles_x;
                    const int tile_x = tile % tiles_x;
                    const int x0 = tile_x * kTileSize;
                    const int y0 = tile_y * kTileSize;
                    const int x1 = std::min(x0 + kTileSize, width);
                    const int y1 = std::min(y0 + kTileSize, height);
                    for (int y = y1 - 1; y >= y0 && !cancel.is_cancelled();
                         --y) {
                        for (int x = x0; x < x1 && !cancel.is_cancelled();
                             ++x) {
                            const std::uint32_t pixel_index =
                                static_cast<std::uint32_t>(y * width + x);
                            for (std::uint32_t sample = 0;
                                 sample < request.samples_per_pixel; ++sample) {
                                if ((sample & 15u) == 0u &&
                                    cancel.is_cancelled()) {
                                    break;
                                }
                                RNG rng(packed_transport::
                                            packed_camera_sample_seed(
                                                request.seed, pixel_index,
                                                sample));
                                const PackedRay ray =
                                    generate_packed_camera_ray(
                                        view.camera, x, y, width, height,
                                        rng);
                                const PackedTransportResult traced =
                                    packed_transport::
                                        trace_packed_path_core_fast(
                                            view, ray, settings, rng);
                                ++stats.samples;
                                stats.traversal_steps +=
                                    traced.traversal_steps;
                                stats.shadow_rays += traced.shadow_rays;
                                Float3 radiance = traced.radiance;
                                if (traced.status !=
                                        PackedTransportStatus::Success ||
                                    !finite(radiance)) {
                                    radiance = {};
                                    ++stats.invalid;
                                } else if (request.sample_clamp > 0.0) {
                                    const double value =
                                        luminance(radiance);
                                    if (value > request.sample_clamp) {
                                        radiance =
                                            packed_transport::math::multiply(
                                                radiance,
                                                static_cast<float>(
                                                    request.sample_clamp /
                                                    value));
                                        ++stats.clamped;
                                    }
                                }
                                film.add_sample(
                                    x, y,
                                    color(radiance.x, radiance.y,
                                          radiance.z));
                            }
                            film.finalize_pixel(x, y);
                        }
                    }
                }
            } catch (...) {
                {
                    std::lock_guard<std::mutex> lock(exception_mutex);
                    if (!first_exception) {
                        first_exception = std::current_exception();
                    }
                }
                failed.store(true, std::memory_order_relaxed);
            }
        };

        std::vector<std::thread> threads;
        threads.reserve(static_cast<std::size_t>(thread_count));
        for (int thread = 0; thread < thread_count; ++thread) {
            threads.emplace_back(worker, thread);
        }
        for (std::thread &thread : threads) {
            thread.join();
        }
        if (first_exception) {
            std::rethrow_exception(first_exception);
        }

        BeautyFilm beauty = film.take_beauty();
        const auto resolve_begin = std::chrono::steady_clock::now();
        RenderBuffer display =
            resolve_beauty(beauty, request.color_pipeline);
        const auto resolve_end = std::chrono::steady_clock::now();
        if (preview != nullptr) {
            preview->publish(display);
        }

        const auto end = std::chrono::steady_clock::now();
        RenderStats stats;
        stats.base.seconds =
            std::chrono::duration<double>(end - begin).count();
        stats.base.compile_seconds = m_preparation.compile_seconds;
        stats.base.scene_bytes = m_preparation.scene_bytes;
        stats.base.resolve_seconds =
            std::chrono::duration<double>(resolve_end - resolve_begin)
                .count();
        stats.base.width = width;
        stats.base.height = height;
        stats.base.samples_per_pixel =
            static_cast<int>(request.samples_per_pixel);
        stats.base.requested_samples =
            static_cast<std::uint64_t>(request.extent.pixel_count()) *
            request.samples_per_pixel;
        stats.base.seed = request.seed;
        stats.cpu.threads = thread_count;
        stats.base.backend = "cpu-packed";
        stats.cuda.device_name = "host";
        for (const WorkerStats &worker_stats_value : worker_stats) {
            stats.base.completed_samples += worker_stats_value.samples;
            stats.base.invalid_samples += worker_stats_value.invalid;
            stats.base.clamped_samples += worker_stats_value.clamped;
            stats.cpu.traversal_steps +=
                worker_stats_value.traversal_steps;
            stats.cpu.shadow_rays += worker_stats_value.shadow_rays;
        }
        stats.base.sample_count = stats.base.completed_samples;
        stats.base.cancelled =
            stats.base.completed_samples < stats.base.requested_samples;

        RenderResult result;
        result.film = std::move(beauty);
        result.display = std::move(display);
        result.stats = std::move(stats);
        return result;
    }

    CompiledScene m_scene;
    CameraConfig m_default_camera;
    double m_time0 = 0.0;
    double m_time1 = 1.0;
    PreparationStats m_preparation;
    std::atomic<bool> m_rendering{false};
};

} // namespace

std::unique_ptr<IRenderSession>
make_cpu_packed_render_session(const SceneIR &ir) {
    return std::make_unique<CpuPackedRenderSession>(ir);
}

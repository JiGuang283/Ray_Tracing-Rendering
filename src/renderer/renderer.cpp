#include "renderer.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "film.h"
#include "sampler.h"
#include "sample_filter.h"

Renderer::Renderer() : m_is_rendering(false) {
}

RenderResult Renderer::render(
    shared_ptr<hittable> world, shared_ptr<camera> cam,
    const color &background, const std::vector<shared_ptr<Light>> &lights,
    const RenderRequest &request, const CancellationToken &cancel,
    PreviewSurface *preview, const LightSampler *provided_light_sampler) {
    validate_render_request(request);
    if (!world || !cam) {
        throw std::invalid_argument("renderer requires a world and camera");
    }
    if (!m_integrator) {
        throw std::invalid_argument("renderer requires an integrator");
    }
    bool expected = false;
    if (!m_is_rendering.compare_exchange_strong(expected, true)) {
        throw std::logic_error("renderer is already rendering");
    }
    struct RenderingGuard {
        std::atomic<bool> &state;
        ~RenderingGuard() {
            state.store(false, std::memory_order_relaxed);
        }
    } rendering_guard{m_is_rendering};

    m_integrator->set_max_depth(static_cast<int>(request.max_depth));

    auto start_time = std::chrono::high_resolution_clock::now();

    const int image_width = static_cast<int>(request.extent.width);
    const int image_height = static_cast<int>(request.extent.height);
    Film film(request.extent, request.color_pipeline, preview);

    constexpr int TILE_SIZE = 16;

    int tiles_x = (image_width + TILE_SIZE - 1) / TILE_SIZE;
    int tiles_y = (image_height + TILE_SIZE - 1) / TILE_SIZE;
    int total_tiles = tiles_x * tiles_y;

    std::atomic<int> next_tile_index(0);

    const int num_threads =
        request.threads > 0
            ? static_cast<int>(request.threads)
            : static_cast<int>(
                  std::max(1u, std::thread::hardware_concurrency()));
    std::vector<std::thread> threads;
    struct WorkerStats {
        std::uint64_t completed_samples = 0;
        std::uint64_t clamped_samples = 0;
        std::uint64_t invalid_samples = 0;
        std::uint64_t traversal_steps = 0;
        std::uint64_t shadow_rays = 0;
    };
    std::vector<WorkerStats> worker_stats(
        static_cast<std::size_t>(num_threads));
    const LightSampler fallback_light_sampler(lights);
    const LightSampler &light_sampler = provided_light_sampler != nullptr
                                            ? *provided_light_sampler
                                            : fallback_light_sampler;
    std::atomic<bool> worker_failed{false};
    std::exception_ptr first_exception;
    std::mutex exception_mutex;

    const auto should_cancel = [&]() {
        return !m_is_rendering.load(std::memory_order_relaxed) ||
               cancel.is_cancelled() ||
               worker_failed.load(std::memory_order_relaxed);
    };

    auto render_worker = [&](int worker_id) {
        try {
            ShaderScratch shader_scratch;
            while (true) {
                int tile_index = next_tile_index.fetch_add(1);
                if (tile_index >= total_tiles || should_cancel()) {
                    break;
                }

                int tile_y = (tiles_y - 1) - tile_index / tiles_x;
                int tile_x = tile_index % tiles_x;

                int x_start = tile_x * TILE_SIZE;
                int y_start = tile_y * TILE_SIZE;
                int x_end = std::min(x_start + TILE_SIZE, image_width);
                int y_end = std::min(y_start + TILE_SIZE, image_height);

                for (int j = y_end - 1; j >= y_start && !should_cancel();
                     j--) {
                    for (int i = x_start; i < x_end && !should_cancel(); i++) {
                        const std::uint32_t pixel_index =
                            static_cast<std::uint32_t>(j * image_width + i);
                        WorkerStats &stats = worker_stats[worker_id];
                        for (std::uint32_t s = 0;
                             s < request.samples_per_pixel; ++s) {
                            if ((s & 15u) == 0u && should_cancel()) {
                                break;
                            }
                            Sampler sampler(render_sample_seed(
                                request.seed, pixel_index, s));
                            IntegratorContext integrator_context{
                                sampler.rng(), shader_scratch, &light_sampler,
                                &stats.traversal_steps,
                                &stats.shadow_rays};
                            CameraSample sample = sampler.next_camera_sample(
                                i, j, image_width, image_height);
                            ray r = cam->get_ray(sample.u, sample.v,
                                                 sampler.rng());
                            const FilteredCameraSample filtered =
                                filter_camera_sample(
                                    m_integrator->Li(r, *world, background,
                                                     integrator_context),
                                    request.sample_clamp);
                            stats.completed_samples++;
                            stats.clamped_samples += filtered.clamped ? 1 : 0;
                            stats.invalid_samples += filtered.invalid ? 1 : 0;
                            film.add_sample(i, j, filtered.radiance);
                        }
                        film.finalize_pixel(i, j);
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
            worker_failed.store(true, std::memory_order_relaxed);
            m_is_rendering.store(false, std::memory_order_relaxed);
        }
    };

    try {
        for (int t = 0; t < num_threads; t++) {
            threads.emplace_back(render_worker, t);
        }
    } catch (...) {
        m_is_rendering.store(false, std::memory_order_relaxed);
        for (auto &thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        throw;
    }

    for (auto &t : threads) {
        t.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    if (first_exception) {
        std::rethrow_exception(first_exception);
    }

    BeautyFilm beauty = film.take_beauty();
    const auto resolve_begin = std::chrono::high_resolution_clock::now();
    RenderBuffer display = resolve_beauty(beauty, request.color_pipeline);
    const auto resolve_end = std::chrono::high_resolution_clock::now();
    if (preview != nullptr) {
        preview->publish(display);
    }
    std::cout << "Rendering finished in " << elapsed.count() << " seconds."
              << std::endl;

    RenderStats stats;
    stats.base.seconds = elapsed.count();
    stats.base.resolve_seconds =
        std::chrono::duration<double>(resolve_end - resolve_begin).count();
    stats.base.width = image_width;
    stats.base.height = image_height;
    stats.base.samples_per_pixel =
        static_cast<int>(request.samples_per_pixel);
    stats.base.requested_samples =
        static_cast<std::uint64_t>(image_width) * image_height *
        request.samples_per_pixel;
    stats.base.seed = request.seed;
    stats.cpu.threads = num_threads;
    for (const WorkerStats &worker : worker_stats) {
        stats.base.completed_samples += worker.completed_samples;
        stats.base.clamped_samples += worker.clamped_samples;
        stats.base.invalid_samples += worker.invalid_samples;
        stats.cpu.traversal_steps += worker.traversal_steps;
        stats.cpu.shadow_rays += worker.shadow_rays;
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

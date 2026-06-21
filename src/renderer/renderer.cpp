#include "renderer.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>

Renderer::Renderer() : m_is_rendering(false) {
}

RenderStats
Renderer::render(shared_ptr<hittable> world, shared_ptr<camera> cam,
                 const color &background, RenderBuffer &target_buffer,
                 const std::vector<shared_ptr<Light>> &lights) {
    m_is_rendering = true;

    auto start_time = std::chrono::high_resolution_clock::now();

    int image_width = target_buffer.width();
    int image_height = target_buffer.height();

    constexpr int TILE_SIZE = 16;

    int tiles_x = (image_width + TILE_SIZE - 1) / TILE_SIZE;
    int tiles_y = (image_height + TILE_SIZE - 1) / TILE_SIZE;
    int total_tiles = tiles_x * tiles_y;

    std::atomic<int> next_tile_index(0);

    const int num_threads =
        m_settings.thread_count > 0
            ? m_settings.thread_count
            : static_cast<int>(
                  std::max(1u, std::thread::hardware_concurrency()));
    std::vector<std::thread> threads;

    auto render_worker = [&](int worker_id) {
        RNG rng(mix_seed(m_settings.seed,
                         static_cast<uint32_t>(worker_id + 1)));
        while (true) {
            int tile_index = next_tile_index.fetch_add(1);
            if (tile_index >= total_tiles) {
                break;
            }
            if (!m_is_rendering) {
                break;
            }

            int tile_y = (tiles_y - 1) - tile_index / tiles_x;
            int tile_x = tile_index % tiles_x;

            int x_start = tile_x * TILE_SIZE;
            int y_start = tile_y * TILE_SIZE;
            int x_end = std::min(x_start + TILE_SIZE, image_width);
            int y_end = std::min(y_start + TILE_SIZE, image_height);

            for (int j = y_end - 1; j >= y_start; j--) {
                for (int i = x_start; i < x_end; i++) {
                    color pixel_color(0, 0, 0);
                    for (int s = 0; s < m_settings.samples_per_pixel; ++s) {
                        auto u = (i + rng.next()) / (image_width - 1);
                        auto v = (j + rng.next()) / (image_height - 1);
                        ray r = cam->get_ray(u, v, rng);
                        if (m_integrator) {
                            pixel_color += m_integrator->Li(
                                r, *world, background, lights, rng);
                        }
                    }
                    write_color_to_buffer(target_buffer, i, j, pixel_color,
                                          m_settings.samples_per_pixel);
                }
            }
        }
    };

    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back(render_worker, t);
    }

    for (auto &t : threads) {
        t.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    m_is_rendering = false;
    std::cout << "Rendering finished in " << elapsed.count() << " seconds."
              << std::endl;

    RenderStats stats;
    stats.seconds = elapsed.count();
    stats.width = image_width;
    stats.height = image_height;
    stats.samples_per_pixel = m_settings.samples_per_pixel;
    stats.sample_count = static_cast<long long>(image_width) * image_height *
                         m_settings.samples_per_pixel;
    stats.seed = m_settings.seed;
    stats.threads = num_threads;
    return stats;
}

void Renderer::write_color_to_buffer(RenderBuffer &buffer, int x, int y,
                                     color pixel_color, int samples) {
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    auto scale = 1.0 / samples;
    r = sqrt(scale * r);
    g = sqrt(scale * g);
    b = sqrt(scale * b);

    buffer.set_pixel(
        x, y,
        color(clamp(r, 0.0, 1.0), clamp(g, 0.0, 1.0), clamp(b, 0.0, 1.0)));
}

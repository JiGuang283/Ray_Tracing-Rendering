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

#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#include "WindowsApp.h"
#include "camera.h"
#include "hittable.h"
#include "material.h"
#include "rtweekend.h"
#include "scenes.h"

namespace RenderConfig {
constexpr int kMaxDepth = 50;
constexpr double kTMin = 0.001;
constexpr double kShutterOpen = 0.0;
constexpr double kShutterClose = 1.0;
} // namespace RenderConfig

static std::vector<std::vector<color>> gCanvas; // Canvas
int gWidth;
int gHeight;
int samples_per_pixel;

shared_ptr<hittable> world;
color background;
unique_ptr<camera> cam;

void rendering();

color ray_color(const ray &r, const color &background, const hittable &world,
                int depth) {
    hit_record rec;

    if (depth <= 0) {
        return color(0, 0, 0);
    }
    if (!world.hit(r, RenderConfig::kTMin, infinity, rec)) {
        return background;
    }

    ray scattered;
    color attenuntion;
    color emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);

    if (!rec.mat_ptr->scatter(r, rec, attenuntion, scattered)) {
        return emitted;
    }

    return emitted +
           attenuntion * ray_color(scattered, background, world, depth - 1);
}

int main(int argc, char *args[]) {

    int scene_id = 10;
    if (argc > 1) {
        scene_id = std::atoi(args[1]);
    }

    SceneConfig config = select_scene(scene_id);

    world = config.world;
    background = config.background;
    gWidth = config.image_width;
    gHeight = static_cast<int>(gWidth / config.aspect_ratio);
    samples_per_pixel = config.samples_per_pixel;

    cam = make_unique<camera>(config.lookfrom, config.lookat, config.vup,
                              config.vfov, config.aspect_ratio, config.aperture,
                              config.focus_dist, RenderConfig::kShutterOpen,
                              RenderConfig::kShutterClose);

    // Create window app handle
    WindowsApp::ptr winApp =
        WindowsApp::getInstance(gWidth, gHeight, "CGAssignment4: Ray Tracing");
    if (winApp == nullptr) {
        std::cerr << "Error: failed to create a window handler" << std::endl;
        return -1;
    }

    // Memory allocation for canvas
    gCanvas.resize(gHeight, std::vector<color>(gWidth));

    // Launch the rendering thread
    // Note: we run the rendering task in another thread to avoid GUI blocking
    std::thread renderingThread(rendering);

    // Window app loop
    while (!winApp->shouldWindowClose()) {
        // Process event
        winApp->processEvent();

        // Display to the screen
        winApp->updateScreenSurface(gCanvas);
    }

    renderingThread.join();

    return 0;
}

void write_color(int x, int y, color pixel_color, int samples_per_color) {
    // Out-of-range detection
    if (x < 0 || x >= gWidth) {
        std::cerr << "Warnning: try to write the pixel out of range: (x,y) -> ("
                  << x << "," << y << ")" << std::endl;
        return;
    }

    if (y < 0 || y >= gHeight) {
        std::cerr << "Warnning: try to write the pixel out of range: (x,y) -> ("
                  << x << "," << y << ")" << std::endl;
        return;
    }

    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    auto scale = 1.0 / samples_per_color;
    r = sqrt(scale * r);
    g = sqrt(scale * g);
    b = sqrt(scale * b);

    // Note: x -> the column number, y -> the row number
    gCanvas[y][x] =
        color(clamp(r, 0.0, 1.0), clamp(g, 0.0, 1.0), clamp(b, 0.0, 1.0));
}

void rendering() {
    // double startFrame = clock();
    auto startFrame = std::chrono::high_resolution_clock::now();

    printf("CGAssignment4 (built %s at %s) \n", __DATE__, __TIME__);
    std::cout << "Ray-tracing based rendering launched..." << std::endl;

    // Image

    const int image_width = gWidth;
    const int image_height = gHeight;

    // Render

    // The main ray-tracing based rendering loop
    // TODO: finish your own ray-tracing renderer according to the given
    // tutorials
    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    std::atomic<int> next_row(image_height - 1);
    auto render_worker = [&]() {
        while (true) {
            int j = next_row.fetch_sub(1);
            if (j < 0)
                break;

            for (int i = 0; i < image_width; i++) {
                color pixel_color(0, 0, 0);
                for (int s = 0; s < samples_per_pixel; ++s) {
                    auto u = (i + random_double()) / (image_width - 1);
                    auto v = (j + random_double()) / (image_height - 1);
                    ray r = cam->get_ray(u, v);
                    pixel_color += ray_color(r, background, *world,
                                             RenderConfig::kMaxDepth);
                }
                write_color(i, j, pixel_color, samples_per_pixel);
            }
        }
    };

    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back(render_worker);
    }

    for (auto &t : threads) {
        t.join();
    }

    auto endFrame = std::chrono::high_resolution_clock::now();
    double timeConsuming =
        std::chrono::duration<double>(endFrame - startFrame).count();

    std::cout << "Ray-tracing based rendering over..." << std::endl;
    std::cout << "The rendering task took " << timeConsuming << " seconds"
              << std::endl;
}
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

#include <array>
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#include "WindowsApp.h"
#include "aarect.h"
#include "box.h"
#include "bvh.h"
#include "camera.h"
#include "constant_medium.h"
#include "hittable_list.h"
#include "material.h"
#include "moving_sphere.h"
#include "rtweekend.h"
#include "sphere.h"

shared_ptr<hittable> random_scene() {
    hittable_list world;

    auto checker = make_shared<checker_texture>(color(0.2, 0.3, 0.1),
                                                color(0.9, 0.9, 0.9));
    world.add(make_shared<sphere>(point3(0, -1000, 0), 1000,
                                  make_shared<lambertian>(checker)));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double();
            point3 center(a + 0.9 * random_double(), 0.2,
                          b + 0.9 * random_double());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                shared_ptr<material> sphere_material;

                if (choose_mat < 0.8) {
                    auto albedo = color::random() * color::random();
                    sphere_material = make_shared<lambertian>(albedo);
                    auto center2 = center + vec3(0, random_double(0, .5), 0);
                    world.add(make_shared<moving_sphere>(
                        center, center2, 0.0, 1.0, 0.2, sphere_material));
                } else if (choose_mat < 0.95) {
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    sphere_material = make_shared<metal>(albedo, fuzz);
                    world.add(
                        make_shared<sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = make_shared<dielectric>(1.5);
    world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

    auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
    world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

    auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

    return make_shared<bvh_node>(world, 0, 1);
}

shared_ptr<hittable> example_light_scene() {
    hittable_list world;

    auto checker = make_shared<checker_texture>(color(0.2, 0.3, 0.1),
                                                color(0.9, 0.9, 0.9));
    world.add(make_shared<sphere>(point3(0, -1000, 0), 1000,
                                  make_shared<lambertian>(checker)));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double();
            point3 center(a + 0.9 * random_double(), 0.2,
                          b + 0.9 * random_double());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                shared_ptr<material> sphere_material;

                if (choose_mat < 0.3) {
                    auto albedo = color::random() * color::random();
                    sphere_material = make_shared<lambertian>(albedo);
                    auto center2 = center + vec3(0, random_double(0, .5), 0);
                    world.add(
                        make_shared<sphere>(center, 0.2, sphere_material));
                } else if (choose_mat < 0.6) {
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    sphere_material = make_shared<metal>(albedo, fuzz);
                    world.add(
                        make_shared<sphere>(center, 0.2, sphere_material));
                } else if (choose_mat < 0.95) {
                    auto difflight =
                        make_shared<diffuse_light>(color::random() * 2);
                    world.add(make_shared<sphere>(center, 0.2, difflight));
                }
            }
        }
    }

    auto material1 = make_shared<dielectric>(1.5);
    world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

    auto material2 = make_shared<diffuse_light>(color(0.4, 0.2, 0.1) * 5);
    world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

    auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

    return make_shared<bvh_node>(world, 0, 1);
}

shared_ptr<hittable> two_spheres() {
    hittable_list objects;

    auto checker = make_shared<checker_texture>(color(0.2, 0.3, 0.1),
                                                color(0.9, 0.9, 0.9));

    objects.add(make_shared<sphere>(point3(0, -10, 0), 10,
                                    make_shared<lambertian>(checker)));
    objects.add(make_shared<sphere>(point3(0, 10, 0), 10,
                                    make_shared<lambertian>(checker)));

    return make_shared<bvh_node>(objects, 0, 1);
}

shared_ptr<hittable> two_perlin_spheres() {
    hittable_list objects;

    auto pertext = make_shared<noise_texture>(4);
    objects.add(make_shared<sphere>(point3(0, -1000, 0), 1000,
                                    make_shared<lambertian>(pertext)));
    objects.add(make_shared<sphere>(point3(0, 2, 0), 2,
                                    make_shared<lambertian>(pertext)));

    return make_shared<bvh_node>(objects, 0, 1);
}

shared_ptr<hittable> earth() {
    auto earth_texture = make_shared<image_texture>("earthmap.jpg");
    auto earth_surface = make_shared<lambertian>(earth_texture);
    auto globe = make_shared<sphere>(point3(0, 0, 0), 2, earth_surface);

    return make_shared<bvh_node>(hittable_list(globe), 0, 1);
}

shared_ptr<hittable> simple_light() {
    hittable_list objects;

    auto pertext = make_shared<lambertian>(color(0.4, 0.6, 0.3));
    objects.add(make_shared<sphere>(point3(0, -1000, 0), 1000, pertext));
    objects.add(make_shared<sphere>(point3(0, 2, 0), 2, pertext));

    auto difflight = make_shared<diffuse_light>(color(4, 4, 4));
    objects.add(make_shared<xy_rect>(3, 5, 1, 3, -2, difflight));
    objects.add(make_shared<sphere>(vec3(0, 7, 0), 2, difflight));

    return make_shared<bvh_node>(objects, 0, 1);
}

shared_ptr<hittable> cornell_box() {
    hittable_list objects;

    auto red = make_shared<lambertian>(color(.65, .05, .05));
    auto white = make_shared<lambertian>(color(.73, .73, .73));
    auto green = make_shared<lambertian>(color(.12, .45, .15));
    auto light = make_shared<diffuse_light>(color(15, 15, 15));

    objects.add(make_shared<yz_rect>(0, 555, 0, 555, 555, green));
    objects.add(make_shared<yz_rect>(0, 555, 0, 555, 0, red));
    objects.add(make_shared<xz_rect>(213, 343, 227, 332, 554, light));
    objects.add(make_shared<xz_rect>(0, 555, 0, 555, 0, white));
    objects.add(make_shared<xz_rect>(0, 555, 0, 555, 555, white));
    objects.add(make_shared<xy_rect>(0, 555, 0, 555, 555, white));

    shared_ptr<hittable> box1 =
        make_shared<box>(point3(0, 0, 0), point3(165, 330, 165), white);
    box1 = make_shared<rotate_y>(box1, 15);
    box1 = make_shared<translate>(box1, vec3(265, 0, 295));
    objects.add(box1);

    shared_ptr<hittable> box2 =
        make_shared<box>(point3(0, 0, 0), point3(165, 165, 165), white);
    box2 = make_shared<rotate_y>(box2, -18);
    box2 = make_shared<translate>(box2, vec3(130, 0, 65));
    objects.add(box2);

    return make_shared<bvh_node>(objects, 0, 1);
}

shared_ptr<hittable> cornell_smoke() {
    hittable_list objects;

    auto red = make_shared<lambertian>(color(.65, .05, .05));
    auto white = make_shared<lambertian>(color(.73, .73, .73));
    auto green = make_shared<lambertian>(color(.12, .45, .15));
    auto light = make_shared<diffuse_light>(color(7, 7, 7));

    objects.add(make_shared<yz_rect>(0, 555, 0, 555, 555, green));
    objects.add(make_shared<yz_rect>(0, 555, 0, 555, 0, red));
    objects.add(make_shared<xz_rect>(113, 443, 127, 432, 554, light));
    objects.add(make_shared<xz_rect>(0, 555, 0, 555, 555, white));
    objects.add(make_shared<xz_rect>(0, 555, 0, 555, 0, white));
    objects.add(make_shared<xy_rect>(0, 555, 0, 555, 555, white));

    shared_ptr<hittable> box1 =
        make_shared<box>(point3(0, 0, 0), point3(165, 330, 165), white);
    box1 = make_shared<rotate_y>(box1, 15);
    box1 = make_shared<translate>(box1, vec3(265, 0, 295));
    box1 = make_shared<constant_medium>(box1, 0.01, color(0, 0, 0));
    objects.add(box1);

    shared_ptr<hittable> box2 =
        make_shared<box>(point3(0, 0, 0), point3(165, 165, 165), white);
    box2 = make_shared<rotate_y>(box2, -18);
    box2 = make_shared<translate>(box2, vec3(130, 0, 65));
    box2 = make_shared<constant_medium>(box2, 0.01, color(1, 1, 1));
    objects.add(box2);

    return make_shared<bvh_node>(objects, 0, 1);
}

shared_ptr<hittable> final_scene() {
    hittable_list boxes1;
    auto ground = make_shared<lambertian>(color(0.48, 0.83, 0.53));

    const int boxes_per_side = 20;
    for (int i = 0; i < boxes_per_side; i++) {
        for (int j = 0; j < boxes_per_side; j++) {
            auto w = 100.0;
            auto x0 = -1000.0 + i * w;
            auto z0 = -1000.0 + j * w;
            auto y0 = 0.0;
            auto x1 = x0 + w;
            auto y1 = random_double(1, 101);
            auto z1 = z0 + w;

            boxes1.add(make_shared<box>(point3(x0, y0, z0), point3(x1, y1, z1),
                                        ground));
        }
    }

    hittable_list objects;

    objects.add(make_shared<bvh_node>(boxes1, 0, 1));

    auto light = make_shared<diffuse_light>(color(7, 7, 7));
    objects.add(make_shared<xz_rect>(123, 423, 147, 412, 554, light));

    auto center1 = point3(400, 400, 200);
    auto center2 = center1 + vec3(30, 0, 0);
    auto moving_sphere_material = make_shared<lambertian>(color(0.7, 0.3, 0.1));
    objects.add(make_shared<moving_sphere>(center1, center2, 0, 1, 50,
                                           moving_sphere_material));

    objects.add(make_shared<sphere>(point3(260, 150, 45), 50,
                                    make_shared<dielectric>(1.5)));
    objects.add(
        make_shared<sphere>(point3(0, 150, 145), 50,
                            make_shared<metal>(color(0.8, 0.8, 0.9), 1.0)));

    auto boundary = make_shared<sphere>(point3(360, 150, 145), 70,
                                        make_shared<dielectric>(1.5));
    objects.add(boundary);
    objects.add(
        make_shared<constant_medium>(boundary, 0.2, color(0.2, 0.4, 0.9)));

    boundary = make_shared<sphere>(point3(0, 0, 0), 5000,
                                   make_shared<dielectric>(1.5));
    objects.add(make_shared<constant_medium>(boundary, .0001, color(1, 1, 1)));

    auto emat =
        make_shared<lambertian>(make_shared<image_texture>("earthmap.jpg"));
    objects.add(make_shared<sphere>(point3(400, 200, 400), 100, emat));

    auto pertext = make_shared<noise_texture>(0.1);
    objects.add(make_shared<sphere>(point3(220, 280, 300), 80,
                                    make_shared<lambertian>(pertext)));

    hittable_list boxes2;
    auto white = make_shared<lambertian>(color(.73, .73, .73));
    int ns = 1000;
    for (int j = 0; j < ns; j++) {
        boxes2.add(make_shared<sphere>(point3::random(0, 165), 10, white));
    }

    objects.add(make_shared<translate>(
        make_shared<rotate_y>(make_shared<bvh_node>(boxes2, 0.0, 1.0), 15),
        vec3(-100, 270, 395)));

    return make_shared<bvh_node>(objects, 0, 1);
}

struct SceneConfig {
    shared_ptr<hittable> world;
    color background;
    point3 lookfrom;
    point3 lookat;
    vec3 vup{0, 1, 0};
    double vfov = 40.0;
    double aperture = 0.0;
    double focus_dist = 10.0;
    double aspect_ratio = 16.0 / 9.0;
    int image_width = 1280;
    int samples_per_pixel = 100;
};

namespace RenderConfig {
constexpr int kMaxDepth = 50;
constexpr double kTMin = 0.001;
constexpr double kShutterOpen = 0.0;
constexpr double kShutterClose = 1.0;
} // namespace RenderConfig

SceneConfig select_scene(int scene_id) {
    SceneConfig config;

    switch (scene_id) {
    case 1:
        config.world = random_scene();
        config.aspect_ratio = 1.0;
        config.image_width = 400;
        config.samples_per_pixel = 50;
        config.background = color(0.70, 0.80, 1.00);
        config.lookfrom = point3(13, 2, 3);
        config.lookat = point3(0, 0, 0);
        config.vfov = 20.0;
        config.aperture = 0.1;
        break;

    case 2:
        config.world = two_spheres();
        config.background = color(0.70, 0.80, 1.00);
        config.lookfrom = point3(13, 2, 3);
        config.lookat = point3(0, 0, 0);
        config.vfov = 20.0;
        break;

    case 4:
        config.world = earth();
        config.lookfrom = point3(13, 2, 3);
        config.background = color(0.70, 0.80, 1.00);
        config.lookat = point3(0, 0, 0);
        config.vfov = 20.0;
        break;

    case 5:
        config.world = simple_light();
        config.background = color(0.0, 0.0, 0.0);
        config.lookfrom = point3(26, 3, 6);
        config.lookat = point3(0, 2, 0);
        config.vfov = 20.0;
        break;

    case 6:
        config.world = example_light_scene();
        config.background = color(0.00, 0.00, 0.00);
        config.lookfrom = point3(13, 2, 3);
        config.lookat = point3(0, 0, 0);
        config.vfov = 20.0;
        config.aperture = 0.0;
        break;

    case 7:
        config.world = cornell_box();
        config.aspect_ratio = 1.0;
        config.image_width = 600;
        config.samples_per_pixel = 40;
        config.background = color(0, 0, 0);
        config.lookfrom = point3(278, 278, -800);
        config.lookat = point3(278, 278, 0);
        config.vfov = 40.0;
        config.aperture = 0.0;
        break;

    case 8:
        config.world = cornell_smoke();
        config.aspect_ratio = 1.0;
        config.image_width = 600;
        config.samples_per_pixel = 200;
        config.background = color(0, 0, 0);
        config.lookfrom = point3(278, 278, -800);
        config.lookat = point3(278, 278, 0);
        config.vfov = 40.0;
        break;

    case 9:
        config.world = final_scene();
        config.aspect_ratio = 1.0;
        config.image_width = 800;
        config.samples_per_pixel = 10000;
        config.background = color(0, 0, 0);
        config.lookfrom = point3(478, 278, -600);
        config.lookat = point3(278, 278, 0);
        config.vfov = 40.0;
        break;

    case 10:
    default:
        config.world = two_perlin_spheres();
        config.background = color(0.70, 0.80, 1.00);
        config.lookfrom = point3(13, 2, 3);
        config.lookat = point3(0, 0, 0);
        config.vfov = 20.0;
        break;
    }

    return config;
}

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
                              config.focus_dist, RenderConfig::kShutterOpen, RenderConfig::kShutterClose);

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
                    pixel_color += ray_color(r, background, *world, RenderConfig::kMaxDepth);
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
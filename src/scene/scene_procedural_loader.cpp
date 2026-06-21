#include "scene_procedural_loader.h"

#include "accel.h"
#include "aarect.h"
#include "box.h"
#include "constant_medium.h"
#include "hittable_list.h"
#include "material.h"
#include "moving_sphere.h"
#include "sphere.h"
#include "texture.h"

namespace scene_loader_internal {

shared_ptr<hittable> build_random_scene_generator(bool emissive_variant) {
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

                if (!emissive_variant) {
                    if (choose_mat < 0.8) {
                        auto albedo = color::random() * color::random();
                        sphere_material = make_shared<lambertian>(albedo);
                        auto center2 =
                            center + vec3(0, random_double(0, .5), 0);
                        world.add(make_shared<moving_sphere>(
                            center, center2, 0.0, 1.0, 0.2,
                            sphere_material));
                    } else if (choose_mat < 0.95) {
                        auto albedo = color::random(0.5, 1);
                        auto fuzz = random_double(0, 0.5);
                        sphere_material = make_shared<metal>(albedo, fuzz);
                        world.add(make_shared<sphere>(center, 0.2,
                                                      sphere_material));
                    }
                } else {
                    if (choose_mat < 0.3) {
                        auto albedo = color::random() * color::random();
                        sphere_material = make_shared<lambertian>(albedo);
                        auto center2 =
                            center + vec3(0, random_double(0, .5), 0);
                        (void)center2;
                        world.add(make_shared<sphere>(center, 0.2,
                                                      sphere_material));
                    } else if (choose_mat < 0.6) {
                        auto albedo = color::random(0.5, 1);
                        auto fuzz = random_double(0, 0.5);
                        sphere_material = make_shared<metal>(albedo, fuzz);
                        world.add(make_shared<sphere>(center, 0.2,
                                                      sphere_material));
                    } else if (choose_mat < 0.95) {
                        auto difflight =
                            make_shared<diffuse_light>(color::random() * 2);
                        world.add(
                            make_shared<sphere>(center, 0.2, difflight));
                    }
                }
            }
        }
    }

    auto material1 = make_shared<dielectric>(1.5);
    world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

    if (emissive_variant) {
        auto material2 = make_shared<diffuse_light>(color(0.4, 0.2, 0.1) * 5);
        world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));
    } else {
        auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
        world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));
    }

    auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

    return make_accel(world, 0, 1);
}

shared_ptr<hittable> build_final_scene_generator(bool nee_variant) {
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

            boxes1.add(make_shared<box>(point3(x0, y0, z0),
                                        point3(x1, y1, z1), ground));
        }
    }

    hittable_list objects;
    objects.add(make_accel(boxes1, 0, 1));

    auto light = make_shared<diffuse_light>(color(7, 7, 7));
    auto light_rect = make_shared<xz_rect>(123, 423, 147, 412, 554, light);
    if (nee_variant) {
        objects.add(make_shared<flip_face>(light_rect));
    } else {
        objects.add(light_rect);
    }

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
        make_shared<rotate_y>(make_accel(boxes2, 0.0, 1.0), 15),
        vec3(-100, 270, 395)));

    return make_accel(objects, 0, 1);
}


} // namespace scene_loader_internal

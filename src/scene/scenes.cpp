#include "scenes.h"
#include "aarect.h"
#include "box.h"
#include "bvh.h"
#include "constant_medium.h"
#include "hittable_list.h"
#include "material.h"
#include "mesh.h"
#include "moving_sphere.h"
#include "sphere.h"
#include "triangle.h"


shared_ptr<hittable> pyramid_pointlight_compare_scene() {
    hittable_list world;

    // Ground
    auto ground_mat = make_shared<lambertian>(color(0.35, 0.35, 0.35));
    world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, ground_mat));

    // "Point light": small emissive sphere (stable & simple)
    // Put it front-right-above to create strong directional shading.
    auto light_mat = make_shared<diffuse_light>(color(60, 60, 60));
    world.add(make_shared<sphere>(point3(3.0, 5.0, 3.5), 0.22, light_mat));

    auto red = make_shared<lambertian>(color(0.85, 0.25, 0.25));

    auto unit_faceN = [](const point3& a, const point3& b, const point3& c) {
        return unit_vector(cross(b - a, c - a));
    };

    // Add a tetrahedron (triangular pyramid) centered near c.
    // smooth=true  -> per-vertex normals (averaged from adjacent face normals)
    // smooth=false -> flat (face normal)
    auto add_tetra = [&](const point3& c, double s, bool smooth) {
        // Geometry: apex + 3 base vertices (tetra-like)
        point3 p0 = c + point3( 0.0,  1.25 * s,  0.0);     // apex
        point3 p1 = c + point3(-1.00 * s, 0.00, -0.85 * s);
        point3 p2 = c + point3( 1.00 * s, 0.00, -0.85 * s);
        point3 p3 = c + point3( 0.00 * s, 0.00,  1.15 * s);

        if (!smooth) {
            // Flat shading
            world.add(make_shared<triangle>(p0, p1, p2, red));
            world.add(make_shared<triangle>(p0, p2, p3, red));
            world.add(make_shared<triangle>(p0, p3, p1, red));
            world.add(make_shared<triangle>(p1, p3, p2, red)); // base
            return;
        }

        // --- Face normals for the 4 faces ---
        vec3 f012 = unit_faceN(p0, p1, p2);
        vec3 f023 = unit_faceN(p0, p2, p3);
        vec3 f031 = unit_faceN(p0, p3, p1);
        vec3 f132 = unit_faceN(p1, p3, p2); // base

        // --- Vertex normals = average of adjacent face normals ---
        // (This produces much more stable smooth shading than "radial normals")
        vec3 n0 = unit_vector(f012 + f023 + f031);
        vec3 n1 = unit_vector(f012 + f031 + f132);
        vec3 n2 = unit_vector(f012 + f023 + f132);
        vec3 n3 = unit_vector(f023 + f031 + f132);

        // Helper: force a vertex normal to be in the same hemisphere as the face normal
        auto hemi = [](const vec3& n, const vec3& face_n) {
            return (dot(n, face_n) < 0) ? (-n) : n;
        };

        // For each face, ensure all its vertex normals point to the same hemisphere as that face.
        // This is crucial to avoid "shadow terminator"-like dark bands on low-poly geometry.
        vec3 n0_012 = hemi(n0, f012), n1_012 = hemi(n1, f012), n2_012 = hemi(n2, f012);
        vec3 n0_023 = hemi(n0, f023), n2_023 = hemi(n2, f023), n3_023 = hemi(n3, f023);
        vec3 n0_031 = hemi(n0, f031), n3_031 = hemi(n3, f031), n1_031 = hemi(n1, f031);
        vec3 n1_132 = hemi(n1, f132), n3_132 = hemi(n3, f132), n2_132 = hemi(n2, f132);

        // Build triangles with per-vertex normals (interpolation ON)
        world.add(make_shared<triangle>(p0, p1, p2, n0_012, n1_012, n2_012, red));
        world.add(make_shared<triangle>(p0, p2, p3, n0_023, n2_023, n3_023, red));
        world.add(make_shared<triangle>(p0, p3, p1, n0_031, n3_031, n1_031, red));
        world.add(make_shared<triangle>(p1, p3, p2, n1_132, n3_132, n2_132, red));
    };

    // Left: smooth
    add_tetra(point3(-1.8, 0.8, 0.0), 1.15, true);

    // Right: flat
    add_tetra(point3( 1.8, 0.8, 0.0), 1.15, false);

    return make_shared<bvh_node>(world, 0, 1);
}



shared_ptr<hittable> triangle_normal_interp_compare_scene() {
    hittable_list world;

    // Ground (darker, to avoid washing out contrast)
    auto ground_mat = make_shared<lambertian>(color(0.35, 0.35, 0.35));
    world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, ground_mat));

    // A small, bright side area light -> strong directionality -> gradient becomes obvious
    // Put it in front-right and slightly above the triangles, so it contributes clear shading.
    auto side_light = make_shared<diffuse_light>(color(35, 35, 35));
    // xy_rect(x0,x1, y0,y1, k=z)
    // This rectangle lies on z = +2.5 plane, facing -Z (by your rect convention).
    world.add(make_shared<xy_rect>(
        1.5,  5.0,   // x range (right side)
        1.2,  4.8,   // y range (above)
        +2.5,
        side_light
    ));

    auto tri_mat = make_shared<lambertian>(color(0.85, 0.25, 0.25));

    // Base triangle (in front of camera, roughly vertical)
    point3 a0(-1.8, 0.8, 0.0);
    point3 a1( 0.2, 0.8, 0.0);
    point3 a2(-0.8, 2.8, 0.0);

    // Deliberately different vertex normals (still mostly "up", but tilted differently)
    vec3 n0 = unit_vector(vec3(-0.8, 1.0,  0.2));
    vec3 n1 = unit_vector(vec3( 0.8, 1.0, -0.2));
    vec3 n2 = unit_vector(vec3( 0.0, 1.0,  1.0));

    // Left: smooth shading (vertex-normal interpolation ON)
    world.add(make_shared<triangle>(
        a0, a1, a2,
        n0, n1, n2,
        tri_mat,
        vec2(0, 0), vec2(0, 0), vec2(0, 0),
        false
    ));

    // Right: same geometry shifted right, but FLAT shading (vertex-normal interpolation OFF)
    vec3 shift(2.6, 0.0, 0.0);
    world.add(make_shared<triangle>(
        a0 + shift, a1 + shift, a2 + shift,
        tri_mat
    ));

    return make_shared<bvh_node>(world, 0, 1);
}


shared_ptr<hittable> triangle_vertex_normal_validation_scene() {
    hittable_list world;

    // Ground
    auto ground_mat = make_shared<lambertian>(color(0.6, 0.6, 0.6));
    world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, ground_mat));

    // A big soft area light above (keeps scene stable)
    auto top_light = make_shared<diffuse_light>(color(6, 6, 6));
    world.add(make_shared<xz_rect>(-8, 8, -8, 8, 6, top_light));

    // A SIDE light that is large enough and (very likely) visible to the camera,
    // making the vertex-normal gradient much easier to observe.
    // xy_rect(x0,x1, y0,y1, k=z, material)
    auto side_light = make_shared<diffuse_light>(color(25, 25, 25));
    world.add(make_shared<xy_rect>(
    2.2,  5.0,    // x 往右移出视野
    2.0,  4.5,    // y 往上移
    +2.0,
    side_light
));




    // Triangle with explicit vertex normals (each vertex normal points differently)
    auto tri_mat = make_shared<lambertian>(color(0.85, 0.25, 0.25));

    point3 v0(-1.5, 0.8, 0.0);
    point3 v1( 1.5, 0.8, 0.0);
    point3 v2( 0.0, 2.8, 0.0);

    // Deliberately different normals to create a visible gradient across the triangle.
    // (We keep them roughly "upwards" so the surface still faces the camera reasonably.)
    vec3 n0 = unit_vector(vec3(-0.3, 1.0,  0.2));
    vec3 n1 = unit_vector(vec3( 0.9, 1.0, -0.1));
    vec3 n2 = unit_vector(vec3( 0.0, 1.0,  1.0));

    // Use the constructor that enables vertex-normal interpolation.
    world.add(make_shared<triangle>(v0, v1, v2,
                                   n0, n1, n2,
                                   tri_mat,
                                   vec2(0, 0), vec2(0, 0), vec2(0, 0),
                                   false));

    return make_shared<bvh_node>(world, 0, 1);
}



shared_ptr<hittable> triangle_hit_validation_scene() {
    hittable_list world;

    // Ground (optional, helps perception)
    auto ground_mat = make_shared<lambertian>(color(0.6, 0.6, 0.6));
    world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, ground_mat));

    // Area light above
    auto light_mat = make_shared<diffuse_light>(color(10, 10, 10));
    world.add(make_shared<xz_rect>(-8, 8, -8, 8, 6, light_mat));

    // One single triangle in front of the camera
    auto tri_mat = make_shared<lambertian>(color(0.85, 0.25, 0.25)); // reddish
    point3 v0(-1.5, 0.8, 0.0);
    point3 v1( 1.5, 0.8, 0.0);
    point3 v2( 0.0, 2.8, 0.0);

    world.add(make_shared<triangle>(v0, v1, v2, tri_mat));

    // Wrap with BVH for consistency (not required, but fine)
    return make_shared<bvh_node>(world, 0, 1);
}

shared_ptr<hittable> triangle_occlusion_validation_scene() {
    hittable_list world;

    auto ground_mat = make_shared<lambertian>(color(0.6, 0.6, 0.6));
    world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, ground_mat));

    auto light_mat = make_shared<diffuse_light>(color(10, 10, 10));
    world.add(make_shared<xz_rect>(-8, 8, -8, 8, 6, light_mat));

    // Triangle (placed slightly farther)
    auto tri_mat = make_shared<lambertian>(color(0.25, 0.35, 0.85)); // bluish
    point3 v0(-1.8, 0.7, -1.0);
    point3 v1( 1.8, 0.7, -1.0);
    point3 v2( 0.0, 3.0, -1.0);
    world.add(make_shared<triangle>(v0, v1, v2, tri_mat));

    // Occluder sphere (closer to camera, should block part of triangle)
    auto occ_mat = make_shared<lambertian>(color(0.85, 0.65, 0.20)); // yellowish
    world.add(make_shared<sphere>(point3(-0.3, 1.6, -0.3), 0.9, occ_mat));

    return make_shared<bvh_node>(world, 0, 1);
}


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

shared_ptr<hittable> pbr_test_scene() {
    hittable_list world;

    auto checker = make_shared<checker_texture>(color(0.2, 0.3, 0.1),
                                                color(0.9, 0.9, 0.9));
    world.add(make_shared<sphere>(point3(0, -1000, 0), 1000,
                                  make_shared<lambertian>(checker)));

    // Left: Gold (Metallic)
    auto gold_albedo = make_shared<solid_color>(0.8, 0.6, 0.2);
    auto gold_rough = make_shared<solid_color>(0.1, 0.1, 0.1);
    auto gold_metal = make_shared<solid_color>(1.0, 1.0, 1.0);
    auto gold_mat =
        make_shared<PBRMaterial>(gold_albedo, gold_rough, gold_metal);
    world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, gold_mat));

    // Middle: Silver/Textured (Metallic with Noise)
    auto noise = make_shared<noise_texture>(4.0);
    auto silver_rough = make_shared<solid_color>(0.2, 0.2, 0.2);
    auto silver_metal = make_shared<solid_color>(1.0, 1.0, 1.0);
    auto mid_mat = make_shared<PBRMaterial>(noise, silver_rough, silver_metal);
    world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, mid_mat));

    // Right: Blue Plastic (Dielectric-like PBR)
    auto blue_albedo = make_shared<solid_color>(0.1, 0.2, 0.5);
    auto blue_rough = make_shared<solid_color>(0.05, 0.05, 0.05);
    auto blue_metal = make_shared<solid_color>(0.0, 0.0, 0.0);
    auto blue_mat =
        make_shared<PBRMaterial>(blue_albedo, blue_rough, blue_metal);
    world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, blue_mat));

    return make_shared<bvh_node>(world, 0, 1);
}

shared_ptr<hittable> pbr_spheres_grid() {
    hittable_list world;

    auto checker = make_shared<checker_texture>(color(0.2, 0.3, 0.1),
                                                color(0.9, 0.9, 0.9));
    world.add(make_shared<sphere>(point3(0, -1000, 0), 1000,
                                  make_shared<lambertian>(checker)));

    int rows = 7;
    int cols = 7;
    double spacing = 2.5;

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            double metallic_val = (double)row / (rows - 1);
            double roughness_val = clamp((double)col / (cols - 1), 0.05, 1.0);

            auto albedo = make_shared<solid_color>(0.5, 0.0, 0.0);
            auto roughness = make_shared<solid_color>(
                roughness_val, roughness_val, roughness_val);
            auto metallic = make_shared<solid_color>(metallic_val, metallic_val,
                                                     metallic_val);

            auto mat = make_shared<PBRMaterial>(albedo, roughness, metallic);

            double x = (col - (cols - 1) / 2.0) * spacing;
            double z = (row - (rows - 1) / 2.0) * spacing;

            world.add(make_shared<sphere>(point3(x, 1, z), 1.0, mat));
        }
    }

    auto light_mat = make_shared<diffuse_light>(color(10, 10, 10));
    // 将主光源移到相机上方 (y=60)，避免遮挡视线
    world.add(make_shared<sphere>(point3(0, 60, 0), 10, light_mat));
    // 调整侧面辅助光源的位置
    world.add(make_shared<sphere>(point3(-20, 10, 20), 2, light_mat));
    world.add(make_shared<sphere>(point3(20, 10, 20), 2, light_mat));

    return make_shared<bvh_node>(world, 0, 1);
}

shared_ptr<hittable> pbr_materials_gallery() {
    hittable_list world;

    auto ground_mat = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, ground_mat));

    // Non-Metals (Metallic = 0.0)
    struct MaterialInfo {
        color albedo;
        double metallic;
        double roughness;
    };

    std::vector<MaterialInfo> non_metals = {
        {color(0.02, 0.02, 0.02), 0.0, 0.5}, // Charcoal
        {color(0.21, 0.28, 0.08), 0.0, 0.5}, // Grass
        {color(0.51, 0.51, 0.51), 0.0, 0.5}, // Concrete
        {color(0.7, 0.7, 0.7), 0.0, 0.5},    // White Paint
        {color(0.81, 0.81, 0.81), 0.0, 0.5}  // Fresh Snow
    };

    std::vector<MaterialInfo> metals = {
        {color(0.54, 0.49, 0.42), 1.0, 0.2}, // Titanium
        {color(0.56, 0.57, 0.58), 1.0, 0.2}, // Iron
        {color(0.95, 0.64, 0.54), 1.0, 0.2}, // Copper
        {color(1.00, 0.71, 0.29), 1.0, 0.2}, // Gold
        {color(0.91, 0.92, 0.92), 1.0, 0.2}, // Aluminium
        {color(0.97, 0.96, 0.91), 1.0, 0.2}  // Silver
    };

    double spacing = 2.5;
    double start_x_nm = -((non_metals.size() - 1) * spacing) / 2.0;

    for (size_t i = 0; i < non_metals.size(); ++i) {
        auto &info = non_metals[i];
        auto mat = make_shared<PBRMaterial>(
            make_shared<solid_color>(info.albedo),
            make_shared<solid_color>(info.roughness, info.roughness,
                                     info.roughness),
            make_shared<solid_color>(info.metallic, info.metallic,
                                     info.metallic));
        world.add(make_shared<sphere>(point3(start_x_nm + i * spacing, 1, -2),
                                      1.0, mat));
    }

    double start_x_m = -((metals.size() - 1) * spacing) / 2.0;
    for (size_t i = 0; i < metals.size(); ++i) {
        auto &info = metals[i];
        auto mat = make_shared<PBRMaterial>(
            make_shared<solid_color>(info.albedo),
            make_shared<solid_color>(info.roughness, info.roughness,
                                     info.roughness),
            make_shared<solid_color>(info.metallic, info.metallic,
                                     info.metallic));
        world.add(make_shared<sphere>(point3(start_x_m + i * spacing, 1, 2),
                                      1.0, mat));
    }

    // Lights
    auto light_mat = make_shared<diffuse_light>(color(10, 10, 10));
    world.add(make_shared<sphere>(point3(0, 20, 10), 5, light_mat));

    return make_shared<bvh_node>(world, 0, 1);
}

shared_ptr<hittable> pbr_reference_scene() {
    hittable_list world;

    auto ground_mat = make_shared<lambertian>(color(0.2, 0.2, 0.2));
    world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, ground_mat));

    struct MaterialInfo {
        color albedo;
        double metallic;
        double roughness;
        const char *name;
    };

    // 1. Metals
    std::vector<MaterialInfo> metals = {
        {color(1.000, 0.766, 0.336), 1.0, 0.2, "Gold"},
        {color(0.955, 0.638, 0.538), 1.0, 0.2, "Copper"},
        {color(0.560, 0.570, 0.580), 1.0, 0.3, "Iron"},
        {color(0.913, 0.922, 0.924), 1.0, 0.1, "Aluminum"}};

    // 2. Non-Metals
    std::vector<MaterialInfo> non_metals = {
        {color(1.0, 0.1, 0.1), 0.0, 0.1, "Red Plastic"},
        {color(0.1, 0.1, 1.0), 0.0, 0.8, "Blue Rubber"},
        {color(1.0, 1.0, 1.0), 0.0, 0.02, "Water/Ice"},
        {color(0.02, 0.02, 0.02), 0.0, 0.9, "Charcoal"},
        {color(0.81, 0.81, 0.81), 0.0, 0.9, "Fresh Snow"}};

    // 3. Roughness Gradient (Gold)
    std::vector<double> roughness_gradient = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0};

    double spacing = 2.5;
    double row_z = 0.0;

    // Row 1: Metals
    row_z = -5.0;
    double start_x = -((metals.size() - 1) * spacing) / 2.0;
    for (size_t i = 0; i < metals.size(); ++i) {
        auto &info = metals[i];
        auto mat = make_shared<PBRMaterial>(
            make_shared<solid_color>(info.albedo),
            make_shared<solid_color>(info.roughness, info.roughness,
                                     info.roughness),
            make_shared<solid_color>(info.metallic, info.metallic,
                                     info.metallic));
        world.add(make_shared<sphere>(point3(start_x + i * spacing, 1, row_z),
                                      1.0, mat));
    }

    // Row 2: Non-Metals
    row_z = 0.0;
    start_x = -((non_metals.size() - 1) * spacing) / 2.0;
    for (size_t i = 0; i < non_metals.size(); ++i) {
        auto &info = non_metals[i];
        auto mat = make_shared<PBRMaterial>(
            make_shared<solid_color>(info.albedo),
            make_shared<solid_color>(info.roughness, info.roughness,
                                     info.roughness),
            make_shared<solid_color>(info.metallic, info.metallic,
                                     info.metallic));
        world.add(make_shared<sphere>(point3(start_x + i * spacing, 1, row_z),
                                      1.0, mat));
    }

    // Row 3: Roughness Gradient (Gold)
    row_z = 5.0;
    start_x = -((roughness_gradient.size() - 1) * spacing) / 2.0;
    color gold_albedo(1.000, 0.766, 0.336);
    for (size_t i = 0; i < roughness_gradient.size(); ++i) {
        double r = roughness_gradient[i];
        auto mat = make_shared<PBRMaterial>(
            make_shared<solid_color>(gold_albedo),
            make_shared<solid_color>(r, r, r),
            make_shared<solid_color>(1.0, 1.0, 1.0)); // Metallic = 1.0
        world.add(make_shared<sphere>(point3(start_x + i * spacing, 1, row_z),
                                      1.0, mat));
    }

    // Lights
    auto light_mat = make_shared<diffuse_light>(color(10, 10, 10));
    world.add(make_shared<sphere>(point3(0, 30, 10), 8, light_mat));
    world.add(make_shared<sphere>(point3(-20, 10, 20), 2, light_mat));
    world.add(make_shared<sphere>(point3(20, 10, 20), 2, light_mat));

    return make_shared<bvh_node>(world, 0, 1);
}

shared_ptr<hittable> point_light_scene() {
    hittable_list world;

    auto ground_mat = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, ground_mat));

    // Diffuse Sphere
    auto lambert = make_shared<lambertian>(color(0.8, 0.2, 0.2));
    world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, lambert));

    // PBR Metal Sphere
    auto albedo = make_shared<solid_color>(0.9, 0.9, 0.9);
    auto roughness = make_shared<solid_color>(0.05, 0.05, 0.05); // Smooth
    auto metallic = make_shared<solid_color>(1.0, 1.0, 1.0);
    auto metal_mat = make_shared<PBRMaterial>(albedo, roughness, metallic);
    world.add(make_shared<sphere>(point3(-3, 1, 0), 1.0, metal_mat));

    // PBR Dielectric-like Sphere
    auto d_albedo = make_shared<solid_color>(0.2, 0.2, 0.8);
    auto d_roughness = make_shared<solid_color>(0.1, 0.1, 0.1);
    auto d_metallic = make_shared<solid_color>(0.0, 0.0, 0.0);
    auto plastic_mat =
        make_shared<PBRMaterial>(d_albedo, d_roughness, d_metallic);
    world.add(make_shared<sphere>(point3(3, 1, 0), 1.0, plastic_mat));

    return make_shared<bvh_node>(world, 0, 1);
}

shared_ptr<hittable> mis_demo() {
    hittable_list world;

    auto ground_mat = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, ground_mat));

    // Smooth Metal (Roughness 0.05) - BSDF sampling dominates for specular reflection
    auto smooth_metal = make_shared<PBRMaterial>(
        make_shared<solid_color>(0.9, 0.9, 0.9),
        make_shared<solid_color>(0.05, 0.05, 0.05),
        make_shared<solid_color>(1.0, 1.0, 1.0)
    );
    world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, smooth_metal));

    // Rough Metal (Roughness 0.5) - NEE dominates
    auto rough_metal = make_shared<PBRMaterial>(
        make_shared<solid_color>(0.9, 0.9, 0.9),
        make_shared<solid_color>(0.5, 0.5, 0.5),
        make_shared<solid_color>(1.0, 1.0, 1.0)
    );
    world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, rough_metal));

    // Diffuse - NEE dominates
    auto diffuse = make_shared<lambertian>(color(0.2, 0.2, 0.8));
    world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, diffuse));

    // Emissive Sphere (BSDF sampling only, as it's not in lights list)
    auto light_mat = make_shared<diffuse_light>(color(10, 5, 5));
    world.add(make_shared<sphere>(point3(0, 1, -3), 1.0, light_mat));

    return make_shared<bvh_node>(world, 0, 1);
}

shared_ptr<hittable> mesh_demo_scene() {
    hittable_list world;

    auto ground_mat = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, ground_mat));

    // materials
    auto matte = make_shared<lambertian>(color(0.8, 0.3, 0.3));                 // red-ish
    auto metal_mat = make_shared<metal>(color(0.8, 0.85, 0.9), 0.05);           // metal-ish

    // ---- Key fix: increase spacing between instances (avoid overlap) ----
    const vec3 scale(1.5, 1.5, 1.5);
    const vec3 left_pos(-5.0, 0.0, 0.0);
    const vec3 mid_pos( 0.0, 0.0, 0.0);
    const vec3 right_pos(5.0, 0.0, 0.0);

    // Left: matte mesh
    auto left_mesh = mesh::load_from_obj("assets/sample_mesh.obj", matte,
                                         left_pos, scale);
    if (left_mesh) {
        world.add(left_mesh);
    }

    // Middle: metal mesh
    auto mid_mesh = mesh::load_from_obj("assets/sample_mesh.obj", metal_mat,
                                        mid_pos, scale);
    if (mid_mesh) {
        world.add(mid_mesh);

        // Right: instanced copy via translate
        world.add(make_shared<translate>(mid_mesh, right_pos - mid_pos));
    }

    return make_shared<bvh_node>(world, 0, 1);
}


struct ModelFeatureSettings {
    bool build_bvh = true;
    bool apply_transform = true;
    bool use_vertex_normals = true;
    bool disable_mesh = false;
    bool duplicate_mesh = true;
};

shared_ptr<hittable>
model_feature_validation_scene(const ModelFeatureSettings &settings) {
    hittable_list world;

    auto ground_mat = make_shared<lambertian>(color(0.6, 0.6, 0.6));
    world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, ground_mat));

    auto area_light = make_shared<diffuse_light>(color(6, 6, 6));
    world.add(make_shared<xz_rect>(-6, 6, -6, 6, 6, area_light));

    auto matte = make_shared<lambertian>(color(0.75, 0.45, 0.35));

    if (!settings.disable_mesh) {
        vec3 translation = settings.apply_transform ? vec3(-2.5, 0.0, 0.0)
                                                    : vec3(0.0, 0.0, 0.0);
        vec3 scale = settings.apply_transform ? vec3(1.5, 1.5, 1.5)
                                              : vec3(1.0, 1.0, 1.0);

        auto main_mesh = mesh::load_from_obj("assets/sample_mesh.obj", matte,
                                             translation, scale, settings.build_bvh,
                                             settings.use_vertex_normals);

        if (main_mesh) {
            world.add(main_mesh);

            if (settings.duplicate_mesh) {
                world.add(make_shared<translate>(main_mesh, vec3(3.5, 0.0, 0.0)));
            }
        }
    } else {
        world.add(make_shared<sphere>(point3(-2.0, 1.0, 0.0), 1.0, matte));
        world.add(make_shared<sphere>(point3(1.5, 1.0, 0.0), 1.0, matte));
    }

    auto mirror = make_shared<metal>(color(0.8, 0.85, 0.9), 0.05);
    world.add(make_shared<sphere>(point3(-4.0, 1.25, -2.0), 1.0, mirror));

    if (settings.build_bvh) {
        return make_shared<bvh_node>(world, 0, 1);
    }

    return make_shared<hittable_list>(world);
}

// === NEW: Mesh BVH stress scene (with smooth normal interpolation) ===
// build_world_bvh: 是否对整个 world 建 BVH（物体级加速）
// build_mesh_bvh:  是否对“单个锥体的三角集合”建 BVH（三角形级加速）
// grid_n:          网格边长（总实例数约 grid_n^2）
shared_ptr<hittable> mesh_bvh_stress_scene(bool build_world_bvh,
                                           bool build_mesh_bvh,
                                           int grid_n) {
    hittable_list world;

    // Ground
    auto ground_mat = make_shared<lambertian>(color(0.55, 0.55, 0.55));
    world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, ground_mat));

    // Large area light above (simple + stable)
    auto light_mat = make_shared<diffuse_light>(color(10, 10, 10));
    world.add(make_shared<xz_rect>(-40, 40, -40, 40, 30, light_mat));

    // ---- Build ONE base pyramid (6 triangles) with per-vertex normal interpolation ----
    auto mesh_mat = make_shared<lambertian>(color(0.75, 0.45, 0.35));

    auto unit_faceN = [](const point3& a, const point3& b, const point3& c) {
        return unit_vector(cross(b - a, c - a));
    };

    // build a square-base pyramid centered at origin (like your sample obj)
    // apex: (0,1,0)
    // base: (-1,0,-1), (1,0,-1), (1,0,1), (-1,0,1)
    auto build_smooth_pyramid = [&](const vec3& base_translation,
                                    const vec3& base_scale,
                                    bool local_build_bvh,
                                    shared_ptr<material> mat) -> shared_ptr<hittable> {

        // apply transform (scale then translate)
        auto T = [&](const point3& p) {
            return point3(p.x() * base_scale.x() + base_translation.x(),
                          p.y() * base_scale.y() + base_translation.y(),
                          p.z() * base_scale.z() + base_translation.z());
        };

        // vertices
        point3 v0 = T(point3( 0, 1, 0));   // apex
        point3 v1 = T(point3(-1, 0,-1));
        point3 v2 = T(point3( 1, 0,-1));
        point3 v3 = T(point3( 1, 0, 1));
        point3 v4 = T(point3(-1, 0, 1));

        // faces (6 triangles)
        // sides:
        // (v0,v2,v1), (v0,v3,v2), (v0,v4,v3), (v0,v1,v4)
        // base (two tris): (v1,v2,v3), (v1,v3,v4)
        vec3 f021 = unit_faceN(v0, v2, v1);
        vec3 f032 = unit_faceN(v0, v3, v2);
        vec3 f043 = unit_faceN(v0, v4, v3);
        vec3 f014 = unit_faceN(v0, v1, v4);
        vec3 f123 = unit_faceN(v1, v2, v3);
        vec3 f134 = unit_faceN(v1, v3, v4);

        // vertex normals = average of adjacent face normals
        // adjacency:
        // v0: all 4 side faces
        // v1: f021, f014, f123, f134
        // v2: f021, f032, f123
        // v3: f032, f043, f123, f134
        // v4: f043, f014, f134
        vec3 n0 = unit_vector(f021 + f032 + f043 + f014);
        vec3 n1 = unit_vector(f021 + f014 + f123 + f134);
        vec3 n2 = unit_vector(f021 + f032 + f123);
        vec3 n3 = unit_vector(f032 + f043 + f123 + f134);
        vec3 n4 = unit_vector(f043 + f014 + f134);

        // force vertex normals to the same hemisphere as the face normal (avoid dark bands)
        auto hemi = [](const vec3& n, const vec3& face_n) {
            return (dot(n, face_n) < 0) ? (-n) : n;
        };

        // build local triangles
        hittable_list local;

        // side 1: (v0,v2,v1)
        local.add(make_shared<triangle>(v0, v2, v1,
            hemi(n0, f021), hemi(n2, f021), hemi(n1, f021), mat));

        // side 2: (v0,v3,v2)
        local.add(make_shared<triangle>(v0, v3, v2,
            hemi(n0, f032), hemi(n3, f032), hemi(n2, f032), mat));

        // side 3: (v0,v4,v3)
        local.add(make_shared<triangle>(v0, v4, v3,
            hemi(n0, f043), hemi(n4, f043), hemi(n3, f043), mat));

        // side 4: (v0,v1,v4)
        local.add(make_shared<triangle>(v0, v1, v4,
            hemi(n0, f014), hemi(n1, f014), hemi(n4, f014), mat));

        // base tri 1: (v1,v2,v3)
        local.add(make_shared<triangle>(v1, v2, v3,
            hemi(n1, f123), hemi(n2, f123), hemi(n3, f123), mat));

        // base tri 2: (v1,v3,v4)
        local.add(make_shared<triangle>(v1, v3, v4,
            hemi(n1, f134), hemi(n3, f134), hemi(n4, f134), mat));

        if (local_build_bvh) {
            return make_shared<bvh_node>(local, 0.0, 1.0);
        }
        return make_shared<hittable_list>(local);
    };

    // Base pyramid: built at origin; instanced via translate (transform)
    auto base_pyramid = build_smooth_pyramid(
        vec3(0.0, 0.0, 0.0),   // translation
        vec3(1.0, 1.0, 1.0),   // scale
        build_mesh_bvh,        // local BVH (triangle-level accel)
        mesh_mat
    );

    // If something went wrong, fall back (shouldn't)
    if (!base_pyramid) {
        auto fallback = make_shared<lambertian>(color(0.4, 0.6, 0.8));
        base_pyramid = make_shared<sphere>(point3(0, 0.5, 0), 0.5, fallback);
    }

    // grid instances
    const double spacing = 2.2;
    const double start = -0.5 * (grid_n - 1) * spacing;

    for (int i = 0; i < grid_n; ++i) {
        for (int j = 0; j < grid_n; ++j) {
            double x = start + i * spacing;
            double z = start + j * spacing;
            // Slightly lift it so it doesn't z-fight with ground
            world.add(make_shared<translate>(base_pyramid, vec3(x, 0.01, z)));
        }
    }

    // One glossy reference sphere (helps visually locate)
    auto mirror = make_shared<metal>(color(0.85, 0.9, 0.95), 0.02);
    world.add(make_shared<sphere>(point3(start - 3.0, 1.0, start - 3.0), 1.0, mirror));

    if (build_world_bvh) {
        return make_shared<bvh_node>(world, 0.0, 1.0);
    }
    return make_shared<hittable_list>(world);
}

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
        config.samples_per_pixel = 400;
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

    case 11:
        config.world = pbr_test_scene();
        config.aspect_ratio = 16.0 / 9.0;
        config.image_width = 800;
        config.samples_per_pixel = 100;
        config.background = color(0.70, 0.80, 1.00);
        config.lookfrom = point3(13, 2, 3);
        config.lookat = point3(0, 0, 0);
        config.vfov = 20.0;
        break;

    case 12:
        config.world = pbr_spheres_grid();
        config.aspect_ratio = 1.0;
        config.image_width = 800;
        config.samples_per_pixel = 500;
        config.background = color(0.05, 0.05, 0.05);
        config.lookfrom = point3(0, 40, 0);
        config.lookat = point3(0, 0, 0);
        config.vup = vec3(0, 0, -1);
        config.vfov = 25.0;
        break;

    case 13:
        config.world = pbr_materials_gallery();
        config.aspect_ratio = 16.0 / 9.0;
        config.image_width = 800;
        config.samples_per_pixel = 1000;
        config.background = color(0.1, 0.1, 0.1);
        config.lookfrom = point3(0, 10, 20);
        config.lookat = point3(0, 0, 0);
        config.vfov = 30.0;
        break;

    case 14:
        config.world = pbr_reference_scene();
        config.aspect_ratio = 16.0 / 9.0;
        config.image_width = 1000;
        config.samples_per_pixel = 5000;
        config.background = color(0.05, 0.05, 0.05);
        config.lookfrom = point3(0, 15, 25);
        config.lookat = point3(0, 0, 0);
        config.vfov = 30.0;
        break;

    case 15:
        config.world = point_light_scene();
        config.aspect_ratio = 16.0 / 9.0;
        config.image_width = 800;
        config.samples_per_pixel = 100;
        config.background = color(0.0, 0.0, 0.0);
        config.lookfrom = point3(0, 5, 10);
        config.lookat = point3(0, 1, 0);
        config.vfov = 30.0;
        config.lights.push_back(
            make_shared<PointLight>(point3(0, 6, 2), color(50, 50, 50)));
        break;

    case 16:
        config.world = mis_demo();
        config.aspect_ratio = 16.0 / 9.0;
        config.image_width = 800;
        config.samples_per_pixel = 500;
        config.background = color(0.1, 0.1, 0.1);
        config.lookfrom = point3(0, 5, 10);
        config.lookat = point3(0, 1, 0);
        config.vfov = 30.0;
        // Add PointLight (NEE sampling)
        config.lights.push_back(
            make_shared<PointLight>(point3(5, 10, 5), color(100, 100, 100)));
        break;

    case 17:
        config.world = mesh_demo_scene();
        config.aspect_ratio = 16.0 / 9.0;
        config.image_width = 800;
        config.samples_per_pixel = 200;
        config.background = color(0.7, 0.8, 1.0);
        config.lookfrom = point3(10, 5, 15);
        config.lookat = point3(0, 1, 0);
        config.vfov = 30.0;
        break;

    case 18: {
        ModelFeatureSettings settings{}; // All features enabled
        config.world = model_feature_validation_scene(settings);
        config.aspect_ratio = 16.0 / 9.0;
        config.image_width = 800;
        config.samples_per_pixel = 10000;
        config.background = color(0.65, 0.75, 0.9);
        config.lookfrom = point3(8, 4, 12);
        config.lookat = point3(0, 1.5, 0);
        config.vfov = 25.0;
        break;
    }

    case 19: {
        ModelFeatureSettings settings{};
        settings.build_bvh = false; // Disable BVH to compare traversal paths
        config.world = model_feature_validation_scene(settings);
        config.aspect_ratio = 16.0 / 9.0;
        config.image_width = 800;
        config.samples_per_pixel = 200;
        config.background = color(0.65, 0.75, 0.9);
        config.lookfrom = point3(8, 4, 12);
        config.lookat = point3(0, 1.5, 0);
        config.vfov = 25.0;
        break;
    }

    case 20: {
        ModelFeatureSettings settings{};
        settings.apply_transform = false; // Render without translate/scale
        config.world = model_feature_validation_scene(settings);
        config.aspect_ratio = 16.0 / 9.0;
        config.image_width = 800;
        config.samples_per_pixel = 200;
        config.background = color(0.65, 0.75, 0.9);
        config.lookfrom = point3(8, 4, 12);
        config.lookat = point3(0, 1.5, 0);
        config.vfov = 25.0;
        break;
    }

    case 21: {
        ModelFeatureSettings settings{};
        settings.use_vertex_normals = false; // Flat shading for comparison
        config.world = model_feature_validation_scene(settings);
        config.aspect_ratio = 16.0 / 9.0;
        config.image_width = 800;
        config.samples_per_pixel = 200;
        config.background = color(0.65, 0.75, 0.9);
        config.lookfrom = point3(8, 4, 12);
        config.lookat = point3(0, 1.5, 0);
        config.vfov = 25.0;
        break;
    }

    case 22: {
        ModelFeatureSettings settings{};
        settings.disable_mesh = true;   // Replace mesh to validate loader path
        settings.duplicate_mesh = false;
        config.world = model_feature_validation_scene(settings);
        config.aspect_ratio = 16.0 / 9.0;
        config.image_width = 800;
        config.samples_per_pixel = 200;
        config.background = color(0.65, 0.75, 0.9);
        config.lookfrom = point3(8, 4, 12);
        config.lookat = point3(0, 1.5, 0);
        config.vfov = 25.0;
        break;
    }

    // ===== NEW stress-test cases =====
    case 23: { // World BVH ON, Mesh BVH ON
        config.world = mesh_bvh_stress_scene(true, true, 15);
        config.aspect_ratio = 16.0 / 9.0;
        config.image_width = 800;
        config.samples_per_pixel = 2000;
        config.background = color(0.65, 0.75, 0.9);
        config.lookfrom = point3(30, 18, 30);
        config.lookat = point3(0, 0.8, 0);
        config.vfov = 35.0;
        break;
    }

    case 24: { // World BVH OFF, Mesh BVH ON
        config.world = mesh_bvh_stress_scene(false, true, 15);
        config.aspect_ratio = 16.0 / 9.0;
        config.image_width = 800;
        config.samples_per_pixel = 2000;
        config.background = color(0.65, 0.75, 0.9);
        config.lookfrom = point3(30, 18, 30);
        config.lookat = point3(0, 0.8, 0);
        config.vfov = 35.0;
        break;
    }

    case 25: { // World BVH ON, Mesh BVH OFF
        config.world = mesh_bvh_stress_scene(true, false, 15);
        config.aspect_ratio = 16.0 / 9.0;
        config.image_width = 800;
        config.samples_per_pixel = 2000;
        config.background = color(0.65, 0.75, 0.9);
        config.lookfrom = point3(30, 18, 30);
        config.lookat = point3(0, 0.8, 0);
        config.vfov = 35.0;
        break;
    }

    case 26: { // World BVH OFF, Mesh BVH OFF (worst-case)
        config.world = mesh_bvh_stress_scene(false, false, 15);
        config.aspect_ratio = 16.0 / 9.0;
        config.image_width = 800;
        config.samples_per_pixel = 2000;
        config.background = color(0.65, 0.75, 0.9);
        config.lookfrom = point3(30, 18, 30);
        config.lookat = point3(0, 0.8, 0);
        config.vfov = 35.0;
        break;
    }

    case 10:
    default:
        config.world = two_perlin_spheres();
        config.background = color(0.70, 0.80, 1.00);
        config.lookfrom = point3(13, 2, 3);
        config.lookat = point3(0, 0, 0);
        config.vfov = 20.0;
        break;

    case 27: {
        config.world = triangle_hit_validation_scene();
        config.aspect_ratio = 16.0 / 9.0;
        config.image_width = 800;
        config.samples_per_pixel = 200;
        config.background = color(0.65, 0.75, 0.9);

        // Camera: look at triangle
        config.lookfrom = point3(0.0, 2.0, 6.5);
        config.lookat   = point3(0.0, 1.8, 0.0);
        config.vfov = 30.0;
        config.aperture = 0.0;
        break;
    }

    case 28: {
        config.world = triangle_occlusion_validation_scene();
        config.aspect_ratio = 16.0 / 9.0;
        config.image_width = 800;
        config.samples_per_pixel = 200;
        config.background = color(0.65, 0.75, 0.9);

        config.lookfrom = point3(0.0, 2.0, 6.5);
        config.lookat   = point3(0.0, 1.8, -0.8);
        config.vfov = 30.0;
        config.aperture = 0.0;
        break;
    }

    case 29: {
    config.world = triangle_vertex_normal_validation_scene();
    config.aspect_ratio = 16.0 / 9.0;
    config.image_width = 800;
    config.samples_per_pixel = 1000; // 稍微高一点，梯度更干净
    config.background = color(0.65, 0.75, 0.9);

    point3 tri_center(0.0, 1.47, 0.0);

    // if light emits toward -Z, reverse direction is +Z,
    // so camera should sit at Z negative and look toward +Z
    config.lookfrom = tri_center + vec3(-1.0, 0.0, -1.0) * 4.5;  // (0, 1.47, -6.5)
    config.lookat   = tri_center;

    config.aperture = 0.0;
    break;
    }

    case 30: {
    config.world = triangle_normal_interp_compare_scene();
    config.aspect_ratio = 16.0 / 9.0;
    config.image_width = 800;
    config.samples_per_pixel = 10000;
    config.background = color(0.65, 0.75, 0.9);

    // Camera: straightforward front view, so differences are easy to see
    config.lookfrom = point3(0.6, 1.75, 4.875);
    config.lookat   = point3(0.6, 1.7, 0.0);
    config.vfov = 30.0;
    config.aperture = 0.0;
    break;
    }

    case 31: {
    config.world = pyramid_pointlight_compare_scene();
    config.aspect_ratio = 16.0 / 9.0;
    config.image_width = 900;
    config.samples_per_pixel = 1500;   // 点光源 + 路径追踪会更噪，建议高一点
    config.background = color(0.65, 0.75, 0.9);

    // Camera: slightly above, looking at center between two pyramids
    config.lookfrom = point3(0.0, 4.4, 14.0);
    config.lookat   = point3(0.0, 1.4, 0.0);
    config.vfov = 28.0;
    config.aperture = 0.0;
    break;
    }



    } 

    return config;
}


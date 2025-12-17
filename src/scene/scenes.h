#ifndef SCENES_H
#define SCENES_H

#include "hittable.h"
#include "point_light.h"
#include "vec3.h"
#include <memory>

using std::shared_ptr;

struct SceneConfig {
    shared_ptr<hittable> world;
    std::vector<shared_ptr<Light>> lights; // 新增光源列表，用于重要性采样
    color background{0, 0, 0};
    point3 lookfrom{13, 2, 3};
    point3 lookat{0, 0, 0};
    vec3 vup{0, 1, 0};
    double vfov = 40.0;
    double aperture = 0.0;
    double focus_dist = 10.0;
    double aspect_ratio = 16.0 / 9.0;
    int image_width = 1280;
    int samples_per_pixel = 100;
};

SceneConfig select_scene(int scene_id);
// === Triangle intersection validation scenes ===
shared_ptr<hittable> pyramid_pointlight_compare_scene();
shared_ptr<hittable> triangle_vertex_normal_validation_scene();
shared_ptr<hittable> triangle_normal_interp_compare_scene();
shared_ptr<hittable> triangle_hit_validation_scene();
shared_ptr<hittable> triangle_occlusion_validation_scene();
shared_ptr<hittable> random_scene();
shared_ptr<hittable> example_light_scene();
shared_ptr<hittable> two_spheres();
shared_ptr<hittable> pbr_test_scene();
shared_ptr<hittable> two_perlin_spheres();
shared_ptr<hittable> earth();
shared_ptr<hittable> simple_light();
shared_ptr<hittable> cornell_box();
shared_ptr<hittable> cornell_smoke();
shared_ptr<hittable> final_scene();
shared_ptr<hittable> pbr_test_scene();
shared_ptr<hittable> pbr_spheres_grid();
shared_ptr<hittable> pbr_materials_gallery();
shared_ptr<hittable> pbr_reference_scene();
shared_ptr<hittable> point_light_scene();
shared_ptr<hittable> mis_demo();
shared_ptr<hittable> mesh_demo_scene();

// === Mesh-focused verification scenes (added for testing) ===
shared_ptr<hittable> mesh_normals_compare_scene();
shared_ptr<hittable> mesh_transform_gallery_scene();
shared_ptr<hittable> mesh_bvh_compare_scene();

// === NEW: Mesh BVH stress scene (large instancing, togglable BVH) ===
shared_ptr<hittable> mesh_bvh_stress_scene(bool build_world_bvh,
                                           bool build_mesh_bvh,
                                           int grid_n = 15);

#endif

#ifndef SCENES_H
#define SCENES_H

#include "scene_config.h"

using std::shared_ptr;

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
shared_ptr<hittable> directional_light_scene();
shared_ptr<hittable> spot_light_scene();
shared_ptr<hittable> environment_light_scene();
shared_ptr<hittable> quad_light_scene();
shared_ptr<hittable> cornell_box_nee();
shared_ptr<hittable> final_scene_nee();
shared_ptr<hittable> mis_demo();
shared_ptr<hittable> mis_comparison_scene();
shared_ptr<hittable> soft_shadow_demo();
shared_ptr<hittable> hdr_demo_scene();

// Final Demo Scenes
shared_ptr<hittable> materials_showcase();
shared_ptr<hittable> cornell_box_extended();
shared_ptr<hittable> interior_lighting_scene();
shared_ptr<hittable> jewelry_display();
shared_ptr<hittable> jewelry_display_simplified();
shared_ptr<hittable> glass_caustics_scene();
shared_ptr<hittable> pbr_texture_demo();
shared_ptr<hittable> pbr_floating_spheres_env();
shared_ptr<hittable> multi_light_demo();

// Fun Demos
shared_ptr<hittable> cmy_shadows_demo();
shared_ptr<hittable> infinity_mirror_demo();

shared_ptr<hittable> pyramid_pointlight_compare_scene();
shared_ptr<hittable> triangle_vertex_normal_validation_scene();
shared_ptr<hittable> triangle_normal_interp_compare_scene();
shared_ptr<hittable> triangle_hit_validation_scene();
shared_ptr<hittable> triangle_occlusion_validation_scene();
shared_ptr<hittable> mesh_demo_scene();
shared_ptr<hittable> mesh_monkey_scene();

#endif

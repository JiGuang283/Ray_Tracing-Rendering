#include "test_harness.h"

#include "bvh.h"
#include "box.h"
#include "hittable_list.h"
#include "material_programs.h"
#include "mesh_asset.h"
#include "mesh_instance.h"
#include "moving_sphere.h"
#include "rtweekend.h"
#include "sphere.h"
#include "transform.h"
#include "transformed_hittable.h"
#include "triangle.h"
#include "aarect.h"

#include <memory>
#include <string>
#include <vector>

namespace {

MaterialHandle white() {
    return make_lambertian_material(color(0.8, 0.8, 0.8));
}

void require_occlusion_matches(const hittable &object, RNG &rng) {
    for (int sample = 0; sample < 500; ++sample) {
        const point3 origin(rng.next_double(-2.0, 2.0),
                            rng.next_double(-2.0, 2.0),
                            rng.next_double(-2.0, 2.0));
        const vec3 direction(rng.next_double(-1.0, 1.0),
                             rng.next_double(-1.0, 1.0),
                             rng.next_double(-1.0, 1.0));
        if (direction.length_squared() <= 1e-12) {
            continue;
        }
        const ray query(origin, unit_vector(direction));
        RNG hit_rng(rng.state);
        RNG occ_rng(rng.state);
        hit_record record;
        const bool hit = object.hit(query, 0.001, 100.0, record, hit_rng);
        const bool occluded = object.occluded(query, 0.001, 100.0, occ_rng);
        REQUIRE(hit == occluded);
    }
}

std::shared_ptr<const MeshAsset> make_test_mesh() {
    std::vector<MeshVertex> vertices = {
        {point3(-1, -1, -4), vec3(0, 0, 1), vec3(1, 0, 0), vec2(0, 0)},
        {point3(1, -1, -4), vec3(0, 0, 1), vec3(1, 0, 0), vec2(1, 0)},
        {point3(1, 1, -4), vec3(0, 0, 1), vec3(1, 0, 0), vec2(1, 1)},
        {point3(-1, 1, -4), vec3(0, 0, 1), vec3(1, 0, 0), vec2(0, 1)},
    };
    const std::uint8_t attributes =
        MESH_ATTRIBUTE_NORMAL | MESH_ATTRIBUTE_UV0;
    std::vector<MeshTriangle> triangles = {
        {{0, 1, 2}, 0, 0, attributes, MESH_TRIANGLE_NONE},
        {{0, 2, 3}, 0, 0, attributes, MESH_TRIANGLE_NONE},
    };
    return std::make_shared<MeshAsset>(
        std::move(vertices), std::move(triangles),
        std::vector<MeshPrimitive>{{"quad", 0, 2, 0}});
}

} // namespace

TEST_CASE(occluded_matches_hit_for_sphere) {
    RNG rng(123);
    require_occlusion_matches(sphere(point3(0, 0, -5), 1.0, white()), rng);
}

TEST_CASE(occluded_matches_hit_for_rect_and_box) {
    RNG rng(124);
    require_occlusion_matches(
        xy_rect(-1.0, 1.0, -1.0, 1.0, -5.0, white()), rng);
    require_occlusion_matches(
        box(point3(-1, -1, -6), point3(1, 1, -4), white()), rng);
}

TEST_CASE(occluded_matches_hit_for_triangle_and_moving_sphere) {
    RNG rng(125);
    const MaterialHandle material = white();
    require_occlusion_matches(
        triangle(point3(-1, -1, -5), point3(1, -1, -5),
                 point3(0, 1, -5), material),
        rng);
    require_occlusion_matches(
        moving_sphere(point3(0, 0, -5), point3(0, 0, -5), 0.0, 1.0, 1.0,
                      material),
        rng);
}

TEST_CASE(occluded_matches_hit_for_transformed_and_flipped) {
    RNG rng(126);
    auto child = std::make_shared<sphere>(point3(0, 0, -5), 1.0, white());
    require_occlusion_matches(
        TransformedHittable(child, Transform::translate(vec3(0.2, 0.0, 0.3))),
        rng);
    require_occlusion_matches(flip_face(child), rng);
}

TEST_CASE(occluded_matches_hit_for_mesh_instance) {
    RNG rng(127);
    const auto asset = make_test_mesh();
    const MeshInstance instance(asset, {white()});
    require_occlusion_matches(instance, rng);
}

TEST_CASE(occluded_matches_hit_for_lists_and_linear_bvh) {
    RNG rng(128);
    hittable_list list;
    list.add(std::make_shared<sphere>(point3(0, 0, -5), 1.0, white()));
    list.add(std::make_shared<xy_rect>(-1.0, 1.0, -1.0, 1.0, -4.5, white()));
    list.add(std::make_shared<triangle>(
        point3(-1, -1, -6), point3(1, -1, -6), point3(0, 1, -6), white()));
    require_occlusion_matches(list, rng);
    require_occlusion_matches(LinearBVH(list, 0.0, 1.0), rng);
}

#include "test_harness.h"

#include "material_programs.h"
#include "mesh_instance.h"
#include "mesh_light.h"

namespace {

std::shared_ptr<const MeshAsset> make_two_triangle_asset(bool build_bvh) {
    std::vector<MeshVertex> vertices(4);
    vertices[0].position = point3(0, 0, 0);
    vertices[1].position = point3(1, 0, 0);
    vertices[2].position = point3(0, 1, 0);
    vertices[3].position = point3(1, 1, 0);
    for (MeshVertex &vertex : vertices) {
        vertex.normal = vec3(0, 0, 1);
    }
    vertices[0].uv0 = vec2(0, 0);
    vertices[1].uv0 = vec2(1, 0);
    vertices[2].uv0 = vec2(0, 1);
    vertices[3].uv0 = vec2(1, 1);

    MeshTriangle first;
    first.vertices[0] = 0;
    first.vertices[1] = 1;
    first.vertices[2] = 2;
    first.material_slot = 0;
    first.attributes = MESH_ATTRIBUTE_NORMAL | MESH_ATTRIBUTE_UV0;
    MeshTriangle second;
    second.vertices[0] = 1;
    second.vertices[1] = 3;
    second.vertices[2] = 2;
    second.material_slot = 1;
    second.attributes = MESH_ATTRIBUTE_NORMAL | MESH_ATTRIBUTE_UV0;
    return std::make_shared<MeshAsset>(
        std::move(vertices), std::vector<MeshTriangle>{first, second},
        std::vector<MeshPrimitive>{{"left", 0, 1, 0},
                                    {"right", 1, 1, 1}},
        build_bvh);
}

class UVEmissionTexture final : public Texture {
  public:
    TextureSample
    evaluate(const ShaderEvalContext &context) const override {
        return TextureSample{color(context.uv0.x(), context.uv0.y(), 0.0),
                             1.0};
    }
};

} // namespace

TEST_CASE(mesh_asset_bvh_matches_brute_force) {
    const auto accelerated = make_two_triangle_asset(true);
    const auto brute_force = make_two_triangle_asset(false);
    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 10; ++x) {
            const ray test_ray(point3((x + 0.5) / 10.0,
                                      (y + 0.5) / 10.0, 1.0),
                               vec3(0, 0, -1));
            MeshIntersection accelerated_hit;
            MeshIntersection brute_hit;
            REQUIRE(accelerated->intersect(test_ray, 0.001, 10.0,
                                           accelerated_hit));
            REQUIRE(brute_force->intersect(test_ray, 0.001, 10.0,
                                           brute_hit));
            REQUIRE_NEAR(accelerated_hit.t, brute_hit.t, 1e-12);
            REQUIRE(accelerated_hit.triangle_index == brute_hit.triangle_index);
        }
    }
}

TEST_CASE(mesh_instances_share_geometry_and_bind_material_slots) {
    const auto asset = make_two_triangle_asset(true);
    const MaterialHandle red =
        make_lambertian_material(color(1.0, 0.0, 0.0));
    const MaterialHandle blue =
        make_lambertian_material(color(0.0, 0.0, 1.0));
    MeshInstance first(asset, {red, blue});
    MeshInstance second(asset, {blue, red},
                        Transform::translate(vec3(2, 0, 0)));
    REQUIRE(first.asset().get() == second.asset().get());

    hit_record left_hit;
    REQUIRE(first.hit(ray(point3(0.2, 0.2, 1), vec3(0, 0, -1)),
                      0.001, 10.0, left_hit));
    REQUIRE(left_hit.material_id == 0);
    REQUIRE(left_hit.mat_ptr == red.get());

    hit_record right_hit;
    REQUIRE(second.hit(ray(point3(2.8, 0.8, 1), vec3(0, 0, -1)),
                       0.001, 10.0, right_hit));
    REQUIRE(right_hit.material_id == 1);
    REQUIRE(right_hit.mat_ptr == red.get());
}

TEST_CASE(mesh_instance_handles_non_uniform_and_mirrored_transforms) {
    const auto asset = make_two_triangle_asset(true);
    const MaterialHandle material =
        make_lambertian_material(color(0.5, 0.5, 0.5));
    MeshInstance instance(
        asset, {material, material},
        Transform::translate(vec3(0, 0, 2)) *
            Transform::scale(vec3(-2, 3, 0.5)));
    hit_record hit;
    REQUIRE(instance.hit(ray(point3(-0.5, 0.5, 4), vec3(0, 0, -1)),
                         0.001, 10.0, hit));
    REQUIRE_NEAR(hit.t, 2.0, 1e-12);
    REQUIRE(dot(hit.normal, hit.geometric_normal) > 0.999);
    REQUIRE(hit.front_face == false);
}

TEST_CASE(mesh_light_evaluates_textured_emission_at_the_sample) {
    std::vector<MeshVertex> vertices(3);
    vertices[0].position = point3(0, 0, 0);
    vertices[1].position = point3(1, 0, 0);
    vertices[2].position = point3(0, 1, 0);
    vertices[0].uv0 = vec2(0, 0);
    vertices[1].uv0 = vec2(1, 0);
    vertices[2].uv0 = vec2(0, 1);
    MeshTriangle triangle;
    triangle.vertices[0] = 0;
    triangle.vertices[1] = 1;
    triangle.vertices[2] = 2;
    triangle.attributes = MESH_ATTRIBUTE_UV0;
    const auto asset = std::make_shared<MeshAsset>(
        std::move(vertices), std::vector<MeshTriangle>{triangle});
    const MaterialHandle material = make_diffuse_light_material(
        std::make_shared<UVEmissionTexture>());
    const auto instance =
        std::make_shared<MeshInstance>(asset,
                                       std::vector<MaterialHandle>{material});
    const MeshLight light(instance, 0);

    const LightSample sample =
        light.sample(point3(0.25, 0.25, 1.0), vec2(0.25, 0.5));
    REQUIRE(sample.pdf > 0.0);
    REQUIRE_NEAR(sample.Li.x(), 0.25, 1e-12);
    REQUIRE_NEAR(sample.Li.y(), 0.25, 1e-12);
}

#include "test_harness.h"

#include "render_data/compiled_scene.h"
#include "render_data/packed_bvh.h"
#include "render_data/resource_compiler.h"
#include "render_data/scene_compiler.h"
#include "material_programs.h"

#include <type_traits>

TEST_CASE(compiled_scene_layout_is_fixed_and_trivially_copyable) {
    REQUIRE(sizeof(PackedBVHNode) == 32);
    REQUIRE(sizeof(PackedTransform) == 96);
    REQUIRE(std::is_trivially_copyable_v<PackedBVHNode>);
    REQUIRE(std::is_trivially_copyable_v<PackedMaterial>);
    REQUIRE(std::is_trivially_copyable_v<PackedLight>);
}

TEST_CASE(compiled_scene_view_borrows_owning_buffers) {
    CompiledScene scene;
    scene.aggregates.push_back({});
    scene.positions.push_back({1.0f, 2.0f, 3.0f, 1.0f});
    scene.normals.push_back({0.0f, 1.0f, 0.0f, 0.0f});
    scene.tangents.push_back({1.0f, 0.0f, 0.0f, 1.0f});
    scene.uv0.push_back({0.25f, 0.75f});
    scene.vertex_colors.push_back({1.0f, 1.0f, 1.0f, 1.0f});

    const CompiledSceneView view = make_scene_view(scene);
    REQUIRE(view.positions.count == 1);
    REQUIRE(view.positions.data == scene.positions.data());
    REQUIRE(view.positions[0].y == 2.0f);
    REQUIRE(validate_compiled_scene(scene).ok());
}

TEST_CASE(compiled_scene_validation_rejects_invalid_ranges) {
    CompiledScene scene;
    scene.aggregates.push_back({});
    PackedMesh mesh;
    mesh.vertices = {1, 1};
    scene.meshes.push_back(mesh);
    REQUIRE(!validate_compiled_scene(scene).ok());
}

TEST_CASE(resource_compiler_deduplicates_topological_texture_graphs) {
    CompiledScene scene;
    PackedResourceCompiler compiler(scene);
    const auto base =
        std::make_shared<SolidColorTexture>(color(0.2, 0.4, 0.6));
    const auto scale = std::make_shared<ScaleTexture>(base, 2.0);
    const auto material = make_lambertian_material(scale);

    const MaterialId first = compiler.compile_material(material);
    const MaterialId second = compiler.compile_material(material);
    REQUIRE(first.value == second.value);
    REQUIRE(scene.materials.size() == 1);
    REQUIRE(scene.texture_nodes.size() == 2);
    REQUIRE(scene.texture_nodes[1].input0 == 0);
    REQUIRE(scene.materials[0].texture_ids[0] == 1);
}

TEST_CASE(resource_compiler_packs_image_and_perlin_storage_once) {
    CompiledScene scene;
    PackedResourceCompiler compiler(scene);
    const auto image = ImageAsset::from_pixels(
        1, 1, 3, std::vector<float>{0.1f, 0.2f, 0.3f});
    const auto texture = std::make_shared<ImageTexture>(image);
    compiler.compile_texture(texture);
    compiler.compile_texture(texture);
    REQUIRE(scene.images.size() == 1);
    REQUIRE(scene.image_texels.size() == 3);

    const auto noise = std::make_shared<NoiseTexture>(4.0);
    compiler.compile_texture(noise);
    REQUIRE(scene.perlin_tables.size() == 1);
    REQUIRE(scene.perlin_gradients.size() == 256);
    REQUIRE(scene.perlin_permutations.size() == 3 * 256);
}

TEST_CASE(packed_bvh_is_preorder_compact_and_reorders_leaf_payloads) {
    std::vector<PackedBVHPrimitive> primitives;
    for (std::uint32_t index = 0; index < 12; ++index) {
        const double x = static_cast<double>(11 - index);
        primitives.push_back(
            {aabb(point3(x, 0, 0), point3(x + 0.25, 1, 1)), index});
    }
    const PackedBVHBuildResult build =
        build_packed_bvh(std::move(primitives));
    REQUIRE(!build.nodes.empty());
    REQUIRE(!build.nodes.front().is_leaf());
    REQUIRE(build.nodes.front().first > 1);
    REQUIRE(build.ordered_payloads.size() == 12);
    REQUIRE(build.max_depth <= 64);
    for (const PackedBVHNode &node : build.nodes) {
        if (node.is_leaf()) {
            REQUIRE(node.primitive_count() <= 4);
        }
        REQUIRE(node.bounds_min.x < node.bounds_max.x);
        REQUIRE(node.bounds_min.y < node.bounds_max.y);
        REQUIRE(node.bounds_min.z < node.bounds_max.z);
    }
}

TEST_CASE(scene_compiler_packs_analytic_and_generated_geometry) {
    SceneIR ir;
    ir.source_path = "test_scene.json";
    ir.name = "packed test";
    ir.textures.push_back(
        {"gray", ConstantTextureIR{color(0.25, 0.5, 0.75)}});
    ir.materials.push_back(
        {"surface", LambertianMaterialIR{TextureIRId{0}}});

    SphereObjectIR sphere;
    sphere.center = point3(-2, 0, 0);
    sphere.radius = 1.0;
    sphere.material = "surface";
    ir.object_nodes.push_back({"sphere", sphere});

    BoxObjectIR box;
    box.minimum = point3(-1, -1, -1);
    box.maximum = point3(1, 1, 1);
    box.material = "surface";
    ir.object_nodes.push_back({"box", box});
    ir.object_nodes.push_back(
        {"transformed box",
         TransformObjectIR{ObjectIRId{1}, Transform::translate(vec3(3, 0, 0))}});
    ir.objects = {0, 2};

    const CompiledScene scene = compile_scene(ir);
    REQUIRE(validate_compiled_scene(scene).ok());
    REQUIRE(scene.aggregates.size() == 1);
    REQUIRE(scene.spheres.size() == 1);
    REQUIRE(scene.meshes.size() == 1);
    REQUIRE(scene.triangles.size() == 12);
    REQUIRE(scene.instances.size() == 2);
    REQUIRE(scene.transforms.size() == 2);
    REQUIRE(scene.materials.size() == 1);
    REQUIRE(scene.texture_nodes.size() == 1);
    REQUIRE(!scene.bvh_nodes.empty());
    REQUIRE(scene.aggregate_instance_indices.size() == 2);
}

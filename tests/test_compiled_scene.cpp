#include "test_harness.h"

#include "render_data/compiled_scene.h"

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

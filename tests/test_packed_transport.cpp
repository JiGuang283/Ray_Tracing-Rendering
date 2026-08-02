#include "test_harness.h"

#include "packed_transport.h"
#include "scene_compiler.h"

#include <cmath>

namespace {

PackedRay test_ray() {
    PackedRay ray;
    ray.origin = {0.0f, 0.0f, -3.0f};
    ray.direction = {0.0f, 0.0f, 1.0f};
    ray.t_min = 0.001f;
    ray.t_max = 1000.0f;
    ray.time = 0.5f;
    return ray;
}

SceneIR make_diffuse_direct_scene() {
    SceneIR ir;
    ir.source_path = "packed_transport_direct.json";
    ir.preset.background = color(0, 0, 0);
    ir.textures.push_back(
        {"albedo", ConstantTextureIR{color(0.8, 0.6, 0.4)}});
    ir.materials.push_back(
        {"surface", LambertianMaterialIR{TextureIRId{0}}});
    SphereObjectIR sphere;
    sphere.radius = 1.0;
    sphere.material = "surface";
    ir.object_nodes.push_back({"sphere", sphere});
    ir.objects = {0};
    ir.lights.push_back(
        {"point", PointLightIR{point3(0, 0, -3), color(4, 4, 4)}});
    return ir;
}

SceneIR make_emissive_scene() {
    SceneIR ir;
    ir.source_path = "packed_transport_emissive.json";
    ir.textures.push_back(
        {"emission", ConstantTextureIR{color(3.0, 2.0, 1.0)}});
    ir.materials.push_back(
        {"light", DiffuseLightMaterialIR{TextureIRId{0}}});
    SphereObjectIR sphere;
    sphere.radius = 1.0;
    sphere.material = "light";
    ir.object_nodes.push_back({"light sphere", sphere});
    ir.objects = {0};
    return ir;
}

bool finite(Float3 value) {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z);
}

} // namespace

TEST_CASE(packed_transport_returns_background_on_miss) {
    SceneIR ir;
    ir.source_path = "packed_transport_background.json";
    ir.preset.background = color(0.2, 0.3, 0.4);
    const CompiledScene scene = compile_scene(ir);
    PackedTransportSettings settings;
    settings.policy = integrator_policy(IntegratorKind::MISPath);
    settings.max_depth = 8;
    RNG rng(123);
    const PackedTransportResult result = trace_packed_path(
        make_scene_view(scene), test_ray(), settings, rng);
    REQUIRE(result.status == PackedTransportStatus::Success);
    REQUIRE_NEAR(result.radiance.x, 0.2, 1e-6);
    REQUIRE_NEAR(result.radiance.y, 0.3, 1e-6);
    REQUIRE_NEAR(result.radiance.z, 0.4, 1e-6);
    REQUIRE(result.depth == 1);
}

TEST_CASE(packed_transport_first_hit_emission_is_unweighted) {
    const CompiledScene scene = compile_scene(make_emissive_scene());
    for (std::uint32_t mode = 0; mode <= 4; ++mode) {
        PackedTransportSettings settings;
        settings.policy = integrator_policy(
            static_cast<IntegratorKind>(mode));
        settings.max_depth = 4;
        RNG rng(19);
        const PackedTransportResult result = trace_packed_path(
            make_scene_view(scene), test_ray(), settings, rng);
        REQUIRE(result.status == PackedTransportStatus::Success);
        REQUIRE_NEAR(result.radiance.x, 3.0, 1e-5);
        REQUIRE_NEAR(result.radiance.y, 2.0, 1e-5);
        REQUIRE_NEAR(result.radiance.z, 1.0, 1e-5);
    }
}

TEST_CASE(packed_direct_transport_evaluates_delta_light_without_selection_pdf) {
    const CompiledScene scene = compile_scene(make_diffuse_direct_scene());
    PackedTransportSettings settings;
    settings.policy = integrator_policy(IntegratorKind::DirectLighting);
    settings.max_depth = 1;
    RNG rng(1234);
    const PackedTransportResult result = trace_packed_path(
        make_scene_view(scene), test_ray(), settings, rng);
    REQUIRE(result.status == PackedTransportStatus::Success);
    REQUIRE_NEAR(result.radiance.x, 0.8 / 3.14159265358979323846, 2e-5);
    REQUIRE_NEAR(result.radiance.y, 0.6 / 3.14159265358979323846, 2e-5);
    REQUIRE_NEAR(result.radiance.z, 0.4 / 3.14159265358979323846, 2e-5);
    REQUIRE(result.shadow_rays == 1);
}

TEST_CASE(packed_transport_is_repeatable_for_fixed_seed) {
    const CompiledScene scene = compile_scene(make_diffuse_direct_scene());
    PackedTransportSettings settings;
    settings.policy = integrator_policy(IntegratorKind::MISPath);
    settings.max_depth = 12;
    RNG first_rng(0x12345678u);
    RNG second_rng(0x12345678u);
    const PackedRay first_ray = generate_packed_camera_ray(
        scene.camera, 4, 3, 9, 7, first_rng);
    const PackedRay second_ray = generate_packed_camera_ray(
        scene.camera, 4, 3, 9, 7, second_rng);
    REQUIRE_NEAR(first_ray.origin.x, second_ray.origin.x, 0.0);
    REQUIRE_NEAR(first_ray.direction.y, second_ray.direction.y, 0.0);
    const PackedTransportResult first = trace_packed_path(
        make_scene_view(scene), first_ray, settings, first_rng);
    const PackedTransportResult second = trace_packed_path(
        make_scene_view(scene), second_ray, settings, second_rng);
    REQUIRE(first.status == second.status);
    REQUIRE_NEAR(first.radiance.x, second.radiance.x, 0.0);
    REQUIRE_NEAR(first.radiance.y, second.radiance.y, 0.0);
    REQUIRE_NEAR(first.radiance.z, second.radiance.z, 0.0);
    REQUIRE(first_rng.state == second_rng.state);
}

TEST_CASE(packed_transport_all_integrator_modes_remain_finite) {
    const CompiledScene scene = compile_scene(make_diffuse_direct_scene());
    const CompiledSceneView view = make_scene_view(scene);
    for (std::uint32_t mode = 0; mode <= 4; ++mode) {
        PackedTransportSettings settings;
        settings.policy = integrator_policy(
            static_cast<IntegratorKind>(mode));
        settings.max_depth = 16;
        for (std::uint32_t sample = 0; sample < 128; ++sample) {
            RNG rng(mix_seed(7123, sample + 1));
            const PackedTransportResult result = trace_packed_path(
                view, test_ray(), settings, rng);
            REQUIRE(result.status == PackedTransportStatus::Success);
            REQUIRE(finite(result.radiance));
            REQUIRE(result.depth <= settings.max_depth);
        }
    }
}

TEST_CASE(packed_transport_rejects_invalid_integrator_and_ray) {
    SceneIR ir;
    ir.source_path = "packed_transport_invalid.json";
    const CompiledScene scene = compile_scene(ir);
    PackedTransportSettings settings;
    settings.policy.kind = static_cast<IntegratorKind>(99);
    RNG rng(1);
    REQUIRE(trace_packed_path(make_scene_view(scene), test_ray(), settings,
                              rng)
                .status == PackedTransportStatus::InvalidInput);
    settings.policy = integrator_policy(IntegratorKind::Path);
    PackedRay ray = test_ray();
    ray.direction = {};
    REQUIRE(trace_packed_path(make_scene_view(scene), ray, settings, rng)
                .status == PackedTransportStatus::InvalidInput);
}

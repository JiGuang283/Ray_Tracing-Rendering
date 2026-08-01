#include "test_harness.h"

#include "render_data/compiled_scene.h"
#include "render_data/flat_intersector.h"
#include "render_data/packed_bvh.h"
#include "render_data/packed_material.h"
#include "render_data/packed_texture.h"
#include "render_data/resource_compiler.h"
#include "render_data/scene_compiler.h"
#include "material_programs.h"

#include <type_traits>

namespace {

SceneIR make_compiler_test_ir() {
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
    return ir;
}

double normal_dot(Float3 packed, const vec3 &reference) {
    return packed.x * reference.x() + packed.y * reference.y() +
           packed.z * reference.z();
}

} // namespace

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

TEST_CASE(packed_texture_evaluator_matches_host_graph) {
    const auto image_asset = ImageAsset::from_pixels(
        2, 2, 4,
        std::vector<float>{
            0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f,
            0.9f, 0.8f, 0.7f, 0.6f, 0.3f, 0.2f, 0.1f, 1.0f});
    SamplerState sampler;
    sampler.wrap_u = WrapMode::Mirror;
    sampler.wrap_v = WrapMode::Clamp;
    const TextureHandle image = std::make_shared<ImageTexture>(
        image_asset, ColorSpace::Linear, sampler, TextureChannel::RGB);
    const TextureHandle transformed = std::make_shared<UVTransformTexture>(
        image, vec2(0.13, -0.07), vec2(1.2, 0.8), 0.35);
    const TextureHandle scaled =
        std::make_shared<ScaleTexture>(transformed, 0.75);
    const TextureHandle vertex = std::make_shared<VertexColorTexture>();
    const TextureHandle multiplied =
        std::make_shared<MultiplyTexture>(scaled, vertex);
    const TextureHandle factor =
        std::make_shared<SolidColorTexture>(color(0.3, 0.3, 0.3));
    const TextureHandle graph = std::make_shared<MixTexture>(
        multiplied, std::make_shared<SolidColorTexture>(color(0.8, 0.1, 0.4)),
        factor);

    ShaderEvalContext host_context;
    host_context.position = point3(0.2, -0.4, 1.3);
    host_context.uv0 = vec2(0.37, 0.62);
    host_context.vertex_color = color(0.6, 0.7, 0.8);
    host_context.vertex_alpha = 0.9;
    const TextureSample expected = graph->evaluate(host_context);

    CompiledScene scene;
    PackedResourceCompiler compiler(scene);
    const TextureId root = compiler.compile_texture(graph);
    const CompiledSceneView view = make_scene_view(scene);
    PackedTextureEvalContext packed_context;
    packed_context.position = {0.2f, -0.4f, 1.3f};
    packed_context.uv0 = {0.37f, 0.62f};
    packed_context.vertex_color = {0.6f, 0.7f, 0.8f, 0.9f};
    Float4 actual;
    REQUIRE(evaluate_packed_texture(view, root.value, packed_context,
                                    actual));
    REQUIRE_NEAR(actual.x, expected.rgb.x(), 1e-5);
    REQUIRE_NEAR(actual.y, expected.rgb.y(), 1e-5);
    REQUIRE_NEAR(actual.z, expected.rgb.z(), 1e-5);
    REQUIRE_NEAR(actual.w, expected.alpha, 1e-5);
}

TEST_CASE(packed_perlin_texture_matches_host_evaluator) {
    const auto noise = std::make_shared<NoiseTexture>(3.25);
    ShaderEvalContext host_context;
    host_context.position = point3(0.31, -1.27, 2.14);
    const TextureSample expected = noise->evaluate(host_context);

    CompiledScene scene;
    PackedResourceCompiler compiler(scene);
    const TextureId root = compiler.compile_texture(noise);
    PackedTextureEvalContext context;
    context.position = {0.31f, -1.27f, 2.14f};
    Float4 actual;
    REQUIRE(evaluate_packed_texture(make_scene_view(scene), root.value,
                                    context, actual));
    REQUIRE_NEAR(actual.x, expected.rgb.x(), 1e-4);
    REQUIRE_NEAR(actual.y, expected.rgb.y(), 1e-4);
    REQUIRE_NEAR(actual.z, expected.rgb.z(), 1e-4);
}

TEST_CASE(packed_material_emission_matches_host_program) {
    const TextureHandle emission = std::make_shared<MultiplyTexture>(
        std::make_shared<SolidColorTexture>(color(2.0, 1.0, 0.5)),
        std::make_shared<VertexColorTexture>());
    const MaterialHandle material = make_diffuse_light_material(emission);
    CompiledScene scene;
    PackedResourceCompiler compiler(scene);
    const MaterialId material_id = compiler.compile_material(material);

    ShaderEvalContext host_context;
    host_context.vertex_color = color(0.25, 0.5, 0.75);
    host_context.vertex_alpha = 0.8;
    ShaderScratch scratch;
    MaterialOutput output;
    material->evaluate(host_context, scratch, output);
    REQUIRE(output.has_emission);

    PackedTextureEvalContext packed_context;
    packed_context.vertex_color = {0.25f, 0.5f, 0.75f, 0.8f};
    Float3 packed_emission;
    REQUIRE(evaluate_packed_material_emission(
        make_scene_view(scene), material_id.value, packed_context,
        packed_emission));
    REQUIRE_NEAR(packed_emission.x, output.emission.x(), 1e-5);
    REQUIRE_NEAR(packed_emission.y, output.emission.y(), 1e-5);
    REQUIRE_NEAR(packed_emission.z, output.emission.z(), 1e-5);
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

TEST_CASE(packed_bounds_round_outward_after_float_conversion) {
    const aabb source(point3(-1.0 / 3.0, -12345.678901, 0.1),
                      point3(2.0 / 3.0, 98765.432109, 10.0 / 7.0));
    const PackedBVHNode packed = pack_packed_bounds(source);
    REQUIRE(static_cast<double>(packed.bounds_min.x) <= source.min().x());
    REQUIRE(static_cast<double>(packed.bounds_min.y) <= source.min().y());
    REQUIRE(static_cast<double>(packed.bounds_min.z) <= source.min().z());
    REQUIRE(static_cast<double>(packed.bounds_max.x) >= source.max().x());
    REQUIRE(static_cast<double>(packed.bounds_max.y) >= source.max().y());
    REQUIRE(static_cast<double>(packed.bounds_max.z) >= source.max().z());
}

TEST_CASE(scene_compiler_packs_analytic_and_generated_geometry) {
    const CompiledScene scene = compile_scene(make_compiler_test_ir());
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

TEST_CASE(scene_compiler_packs_explicit_and_automatic_lights) {
    SceneIR ir;
    ir.source_path = "light_test.json";
    ir.textures.push_back(
        {"emission", ConstantTextureIR{color(8.0, 4.0, 2.0)}});
    ir.materials.push_back(
        {"light", DiffuseLightMaterialIR{TextureIRId{0}}});

    SphereObjectIR sphere;
    sphere.center = point3(0, 1, 0);
    sphere.radius = 0.5;
    sphere.material = "light";
    ir.object_nodes.push_back({"emissive sphere", sphere});
    QuadObjectIR quad;
    quad.origin = point3(-1, 3, -1);
    quad.u = vec3(2, 0, 0);
    quad.v = vec3(0, 0, 2);
    quad.material = "light";
    ir.object_nodes.push_back({"emissive quad", quad});
    ir.objects = {0, 1};

    ir.lights.push_back(
        {"point", PointLightIR{point3(1, 2, 3), color(3, 2, 1)}});
    ir.lights.push_back({"directional",
                         DirectionalLightIR{vec3(0, -1, 0), color(1, 1, 1)}});
    SpotLightIR spot;
    spot.position = point3(-1, 2, 0);
    spot.direction = vec3(0, -1, 0);
    spot.cutoff = 25.0;
    spot.intensity = color(2, 3, 4);
    ir.lights.push_back({"spot", spot});
    ir.lights.push_back(
        {"quad", QuadLightIR{point3(-1, 4, -1), vec3(2, 0, 0),
                              vec3(0, 0, 2), color(4, 4, 4)}});

    const CompiledScene scene = compile_scene(ir);
    REQUIRE(scene.lights.size() == 6);
    REQUIRE(scene.delta_light_indices.size() == 3);
    REQUIRE(scene.non_delta_light_indices.size() == 3);
    REQUIRE(scene.light_selection_probabilities.size() == 3);
    REQUIRE_NEAR(scene.light_cdf.back(), 1.0, 1e-6);
    for (float probability : scene.light_selection_probabilities) {
        REQUIRE(probability >= 0.05f / 3.0f);
    }
    REQUIRE(scene.lights[4].type == PackedLightType::SphereEmitter);
    REQUIRE(scene.lights[5].type == PackedLightType::MeshEmitter);
    REQUIRE(scene.lights[5].element_indices.count == 2);
    REQUIRE(scene.lights[5].distribution.count == 4);
    REQUIRE((scene.lights[5].flags & PACKED_LIGHT_BSDF_HITTABLE) != 0);
    REQUIRE(validate_compiled_scene(scene).ok());
}

TEST_CASE(scene_compiler_builds_environment_distribution) {
    SceneIR ir;
    ir.source_path = "environment_test.json";
    ir.auto_emitters = false;
    ir.lights.push_back(
        {"environment", EnvironmentLightIR{"missing_environment.hdr"}});
    const CompiledScene scene = compile_scene(ir);
    REQUIRE(scene.lights.size() == 1);
    REQUIRE(scene.images.size() == 1);
    REQUIRE(scene.lights[0].type == PackedLightType::Environment);
    REQUIRE((scene.lights[0].flags & PACKED_LIGHT_INFINITE) != 0);
    REQUIRE((scene.lights[0].flags & PACKED_LIGHT_BSDF_HITTABLE) != 0);
    REQUIRE(scene.lights[0].distribution.count == 15);
    REQUIRE(scene.non_delta_light_indices.size() == 1);
    REQUIRE_NEAR(scene.light_selection_probabilities[0], 1.0, 1e-6);
    REQUIRE_NEAR(scene.light_cdf[0], 1.0, 1e-6);
}

TEST_CASE(flat_intersector_hits_and_reconstructs_sphere_and_mesh) {
    const CompiledScene scene = compile_scene(make_compiler_test_ir());
    const CompiledSceneView view = make_scene_view(scene);

    PackedRay sphere_ray;
    sphere_ray.origin = {-2.0f, 0.0f, -3.0f};
    sphere_ray.direction = {0.0f, 0.0f, 1.0f};
    sphere_ray.t_min = 0.001f;
    sphere_ray.t_max = 100.0f;
    PackedHit sphere_hit;
    REQUIRE(intersect_compiled_scene(view, sphere_ray, sphere_hit));
    REQUIRE((sphere_hit.flags & PACKED_HIT_SPHERE) != 0);
    REQUIRE_NEAR(sphere_hit.t, 2.0, 1e-5);
    REQUIRE(sphere_hit.material_id == 0);

    PackedSurfaceInteraction sphere_surface;
    REQUIRE(reconstruct_compiled_hit(view, sphere_ray, sphere_hit,
                                     sphere_surface));
    REQUIRE_NEAR(sphere_surface.position.z, -1.0, 1e-5);
    REQUIRE_NEAR(sphere_surface.geometric_normal.z, -1.0, 1e-5);
    REQUIRE_NEAR(sphere_surface.uv.x, 0.75, 1e-5);
    REQUIRE_NEAR(sphere_surface.uv.y, 0.5, 1e-5);

    PackedRay box_ray;
    box_ray.origin = {3.0f, 0.0f, -3.0f};
    box_ray.direction = {0.0f, 0.0f, 1.0f};
    box_ray.t_min = 0.001f;
    box_ray.t_max = 100.0f;
    PackedHit box_hit;
    REQUIRE(intersect_compiled_scene(view, box_ray, box_hit));
    REQUIRE((box_hit.flags & PACKED_HIT_TRIANGLE) != 0);
    REQUIRE_NEAR(box_hit.t, 2.0, 1e-5);
    REQUIRE(box_hit.material_id == 0);

    PackedSurfaceInteraction box_surface;
    REQUIRE(reconstruct_compiled_hit(view, box_ray, box_hit, box_surface));
    REQUIRE_NEAR(box_surface.position.x, 3.0, 1e-5);
    REQUIRE_NEAR(box_surface.position.z, -1.0, 1e-5);
    REQUIRE(std::abs(box_surface.geometric_normal.z) > 0.999f);
}

TEST_CASE(flat_intersector_matches_reference_geometry_hits) {
    const SceneIR ir = make_compiler_test_ir();
    const SceneConfig reference = build_scene_config(ir);
    const CompiledScene packed = compile_scene(ir);
    const CompiledSceneView view = make_scene_view(packed);
    const std::array<point3, 4> origins{
        point3(-2, 0, -3), point3(-2, 0.5, -3),
        point3(3, 0, -3), point3(10, 0, -3)};

    for (const point3 &origin : origins) {
        const ray reference_ray(origin, vec3(0, 0, 1));
        hit_record reference_hit;
        RNG reference_rng(123);
        const bool reference_found = reference.scene.world->hit(
            reference_ray, 0.001, 100.0, reference_hit, reference_rng);

        PackedRay packed_ray;
        packed_ray.origin = {static_cast<float>(origin.x()),
                             static_cast<float>(origin.y()),
                             static_cast<float>(origin.z())};
        packed_ray.direction = {0.0f, 0.0f, 1.0f};
        packed_ray.t_min = 0.001f;
        packed_ray.t_max = 100.0f;
        PackedHit packed_hit;
        RNG packed_rng(123);
        const bool packed_found = intersect_compiled_scene(
            view, packed_ray, packed_hit, &packed_rng);
        REQUIRE(packed_found == reference_found);
        if (!packed_found) {
            continue;
        }
        REQUIRE_NEAR(packed_hit.t, reference_hit.t, 5e-4);
        PackedSurfaceInteraction surface;
        REQUIRE(reconstruct_compiled_hit(view, packed_ray, packed_hit,
                                         surface));
        REQUIRE_NEAR(surface.uv.x, reference_hit.u, 5e-4);
        REQUIRE_NEAR(surface.uv.y, reference_hit.v, 5e-4);
        const double normal_dot =
            surface.geometric_normal.x * reference_hit.geometric_normal.x() +
            surface.geometric_normal.y * reference_hit.geometric_normal.y() +
            surface.geometric_normal.z * reference_hit.geometric_normal.z();
        REQUIRE(normal_dot >= 0.999);
    }
}

TEST_CASE(flat_intersector_samples_medium_only_with_explicit_rng) {
    SceneIR ir;
    ir.source_path = "medium_test.json";
    SphereObjectIR boundary;
    boundary.center = point3(0, 0, 0);
    boundary.radius = 1.0;
    boundary.material = "boundary";
    ir.object_nodes.push_back({"boundary", boundary});
    ConstantMediumObjectIR medium;
    medium.boundary = 0;
    medium.density = 1000.0;
    medium.albedo = color(0.5, 0.6, 0.7);
    ir.object_nodes.push_back({"medium", medium});
    ir.objects = {1};
    ir.textures.push_back(
        {"boundary_color", ConstantTextureIR{color(0.5, 0.5, 0.5)}});
    ir.materials.push_back(
        {"boundary", LambertianMaterialIR{TextureIRId{0}}});

    const CompiledScene packed = compile_scene(ir);
    const CompiledSceneView view = make_scene_view(packed);
    PackedRay ray;
    ray.origin = {0.0f, 0.0f, -3.0f};
    ray.direction = {0.0f, 0.0f, 1.0f};
    ray.t_min = 0.001f;
    ray.t_max = 100.0f;
    PackedHit hit;
    REQUIRE(!intersect_compiled_scene(view, ray, hit));

    RNG rng(123);
    REQUIRE(intersect_compiled_scene(view, ray, hit, &rng));
    REQUIRE((hit.flags & PACKED_HIT_MEDIUM) != 0);
    REQUIRE(hit.t > 2.0f);
    REQUIRE(hit.t < 4.0f);
    PackedSurfaceInteraction surface;
    REQUIRE(reconstruct_compiled_hit(view, ray, hit, surface));
    REQUIRE(surface.material_id == hit.material_id);
}

TEST_CASE(flat_intersector_preserves_negative_sphere_and_mirrored_transform) {
    SceneIR ir;
    ir.source_path = "negative_sphere_test.json";
    ir.textures.push_back(
        {"surface_color", ConstantTextureIR{color(0.4, 0.5, 0.6)}});
    ir.materials.push_back(
        {"surface", LambertianMaterialIR{TextureIRId{0}}});
    SphereObjectIR sphere;
    sphere.radius = -1.0;
    sphere.material = "surface";
    ir.object_nodes.push_back({"negative sphere", sphere});
    ir.object_nodes.push_back(
        {"transformed sphere",
         TransformObjectIR{
             ObjectIRId{0}, Transform::scale(vec3(-2.0, 1.0, 0.5))}});
    ir.objects = {1};

    const SceneConfig reference = build_scene_config(ir);
    const CompiledScene packed = compile_scene(ir);
    const ray reference_ray(point3(0, 0, -3), vec3(0, 0, 1));
    hit_record reference_hit;
    REQUIRE(reference.scene.world->hit(reference_ray, 0.001, 100.0,
                                       reference_hit));
    PackedRay packed_ray;
    packed_ray.origin = {0.0f, 0.0f, -3.0f};
    packed_ray.direction = {0.0f, 0.0f, 1.0f};
    packed_ray.t_min = 0.001f;
    packed_ray.t_max = 100.0f;
    PackedHit packed_hit;
    REQUIRE(intersect_compiled_scene(make_scene_view(packed), packed_ray,
                                     packed_hit));
    REQUIRE_NEAR(packed_hit.t, reference_hit.t, 5e-4);
    PackedSurfaceInteraction surface;
    REQUIRE(reconstruct_compiled_hit(make_scene_view(packed), packed_ray,
                                     packed_hit, surface));
    REQUIRE(normal_dot(surface.geometric_normal,
                       reference_hit.geometric_normal) >= 0.999);
    REQUIRE_NEAR(surface.uv.x, reference_hit.u, 5e-4);
    REQUIRE_NEAR(surface.uv.y, reference_hit.v, 5e-4);
    REQUIRE(((surface.flags & PACKED_HIT_FRONT_FACE) != 0) ==
            reference_hit.front_face);
}

TEST_CASE(flat_medium_free_path_matches_exponential_mean) {
    SceneIR ir;
    ir.source_path = "medium_statistics_test.json";
    ir.textures.push_back(
        {"boundary_color", ConstantTextureIR{color(0.5, 0.5, 0.5)}});
    ir.materials.push_back(
        {"boundary", LambertianMaterialIR{TextureIRId{0}}});
    SphereObjectIR boundary;
    boundary.radius = 1.0;
    boundary.material = "boundary";
    ir.object_nodes.push_back({"boundary", boundary});
    ConstantMediumObjectIR medium;
    medium.boundary = 0;
    medium.density = 10.0;
    ir.object_nodes.push_back({"medium", medium});
    ir.objects = {1};
    const CompiledScene packed = compile_scene(ir);
    const CompiledSceneView view = make_scene_view(packed);
    PackedRay ray;
    ray.origin = {0.0f, 0.0f, -3.0f};
    ray.direction = {0.0f, 0.0f, 1.0f};
    ray.t_min = 0.001f;
    ray.t_max = 100.0f;

    constexpr int sample_count = 50000;
    RNG rng(9876);
    double sum = 0.0;
    int hits = 0;
    for (int sample = 0; sample < sample_count; ++sample) {
        PackedHit hit;
        if (intersect_compiled_scene(view, ray, hit, &rng)) {
            sum += hit.t - 2.0;
            ++hits;
        }
    }
    REQUIRE(hits == sample_count);
    REQUIRE_NEAR(sum / hits, 0.1, 0.002);
}

#include "test_harness.h"

#include "packed_light.h"
#include "scene_compiler.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

SceneIR make_light_scene() {
    SceneIR ir;
    ir.source_path = "packed_light_test.json";
    ir.textures.push_back(
        {"emission", ConstantTextureIR{color(8.0, 4.0, 2.0)}});
    ir.materials.push_back(
        {"light", DiffuseLightMaterialIR{TextureIRId{0}}});

    SphereObjectIR sphere;
    sphere.center = point3(0, 1, 0);
    sphere.radius = 0.5;
    sphere.material = "light";
    ir.object_nodes.push_back({"sphere", sphere});
    ir.object_nodes.push_back(
        {"scaled sphere",
         TransformObjectIR{ObjectIRId{0},
                           Transform::scale(vec3(2.0, 1.0, 0.5))}});

    QuadObjectIR quad;
    quad.origin = point3(-1, -1, 2);
    quad.u = vec3(2, 0, 0);
    quad.v = vec3(0.5, 2, 0);
    quad.material = "light";
    ir.object_nodes.push_back({"quad", quad});
    ir.objects = {1, 2};

    ir.lights.push_back(
        {"point", PointLightIR{point3(1, 2, 3), color(3, 2, 1)}});
    ir.lights.push_back({
        "directional",
        DirectionalLightIR{vec3(0, -1, 0), color(0.25, 0.5, 1.0)}});
    SpotLightIR spot;
    spot.position = point3(-1, 2, 0);
    spot.direction = vec3(0, -1, 0);
    spot.cutoff = 25.0;
    spot.intensity = color(2, 3, 4);
    ir.lights.push_back({"spot", spot});
    ir.lights.push_back(
        {"quad light",
         QuadLightIR{point3(-1, -1, 4), vec3(2, 0, 0),
                     vec3(0.5, 2, 0), color(4, 4, 4)}});
    return ir;
}

std::uint32_t find_light(const CompiledScene &scene, PackedLightType type) {
    for (std::uint32_t index = 0; index < scene.lights.size(); ++index) {
        if (scene.lights[index].type == type) {
            return index;
        }
    }
    return kInvalidPackedIndex;
}

bool nearly_equal(Float3 a, Float3 b, float tolerance = 1e-4f) {
    return std::abs(a.x - b.x) <= tolerance &&
           std::abs(a.y - b.y) <= tolerance &&
           std::abs(a.z - b.z) <= tolerance;
}

CompiledScene make_constant_environment() {
    CompiledScene scene;
    scene.aggregates.push_back({});
    PackedImageDesc image;
    image.width = 4;
    image.height = 2;
    image.channels = 3;
    image.texels = {0, 24};
    scene.images.push_back(image);
    scene.image_texels.assign(24, 1.0f);

    PackedLight light;
    light.type = PackedLightType::Environment;
    light.flags = PACKED_LIGHT_INFINITE | PACKED_LIGHT_BSDF_HITTABLE;
    light.image_id = 0;
    light.selection_probability = 1.0f;
    light.data0 = {4.0f, 2.0f, 0.0f, 0.0f};
    light.distribution = {0, 23};
    scene.lights.push_back(light);
    scene.non_delta_light_indices.push_back(0);
    scene.light_selection_probabilities.push_back(1.0f);
    scene.light_cdf.push_back(1.0f);

    for (int index = 0; index < 8; ++index) {
        scene.light_distributions.push_back(0.25f);
    }
    for (int row = 0; row < 2; ++row) {
        scene.light_distributions.insert(
            scene.light_distributions.end(),
            {0.0f, 0.25f, 0.5f, 0.75f, 1.0f});
    }
    scene.light_distributions.insert(scene.light_distributions.end(),
                                     {0.5f, 0.5f});
    scene.light_distributions.insert(scene.light_distributions.end(),
                                     {0.0f, 0.5f, 1.0f});
    return scene;
}

} // namespace

TEST_CASE(packed_light_compiler_binds_emitters_to_material_slots) {
    const CompiledScene scene = compile_scene(make_light_scene());
    REQUIRE(validate_compiled_scene(scene).ok());
    REQUIRE(scene.emitter_bindings.size() == scene.material_bindings.size());
    REQUIRE(scene.instances.size() == 2);
    REQUIRE(scene.emitter_bindings[
                scene.instances[0].material_bindings.offset] !=
            kInvalidPackedIndex);
    REQUIRE(scene.emitter_bindings[
                scene.instances[1].material_bindings.offset] !=
            kInvalidPackedIndex);
    for (std::size_t index = 0;
         index < scene.non_delta_light_indices.size(); ++index) {
        const std::uint32_t light_id =
            scene.non_delta_light_indices[index];
        REQUIRE_NEAR(scene.lights[light_id].selection_probability,
                     scene.light_selection_probabilities[index],
                     1e-6);
    }
}

TEST_CASE(packed_delta_lights_are_deterministic_and_finite) {
    const CompiledScene scene = compile_scene(make_light_scene());
    const CompiledSceneView view = make_scene_view(scene);

    PackedLightSample point;
    REQUIRE(sample_packed_light(view, find_light(scene, PackedLightType::Point),
                                {1, 2, 1}, {0.1f, 0.9f}, point) ==
            PackedLightStatus::Success);
    REQUIRE(point.is_delta());
    REQUIRE_NEAR(point.distance, 2.0, 1e-6);
    REQUIRE_NEAR(point.radiance.x, 0.75, 1e-6);
    REQUIRE_NEAR(point.radiance.y, 0.5, 1e-6);
    REQUIRE_NEAR(point.radiance.z, 0.25, 1e-6);

    PackedLightSample directional;
    REQUIRE(sample_packed_light(
                view, find_light(scene, PackedLightType::Directional),
                {0, 0, 0}, {0.8f, 0.2f}, directional) ==
            PackedLightStatus::Success);
    REQUIRE(directional.is_delta());
    REQUIRE(directional.is_infinite() == false);
    REQUIRE_NEAR(directional.wi.y, 1.0, 1e-6);
    REQUIRE_NEAR(directional.radiance.z, 1.0, 1e-6);

    PackedLightSample spot;
    REQUIRE(sample_packed_light(view, find_light(scene, PackedLightType::Spot),
                                {-1, 0, 0}, {0.4f, 0.7f}, spot) ==
            PackedLightStatus::Success);
    REQUIRE(spot.radiance.x > 0.0f);
}

TEST_CASE(packed_skew_quad_sample_and_pdf_are_consistent) {
    CompiledScene scene;
    scene.aggregates.push_back({});
    PackedLight light;
    light.type = PackedLightType::Quad;
    light.data0 = {-1.0f, -1.0f, 2.0f, 1.0f};
    light.data1 = {2.0f, 0.0f, 0.0f, 0.0f};
    light.data2 = {0.5f, 2.0f, 0.0f, 0.0f};
    light.radiance = {3.0f, 2.0f, 1.0f, 0.0f};
    scene.lights.push_back(light);
    const CompiledSceneView view = make_scene_view(scene);

    PackedLightSample sample;
    const Float3 origin{0, 0, 4};
    REQUIRE(sample_packed_light(view, 0, origin, {0.37f, 0.61f}, sample) ==
            PackedLightStatus::Success);
    float pdf = 0.0f;
    REQUIRE(evaluate_packed_light_pdf(view, 0, origin, sample.wi, pdf) ==
            PackedLightStatus::Success);
    REQUIRE_NEAR(pdf, sample.pdf, 2e-5);
    REQUIRE(nearly_equal(sample.radiance, {3, 2, 1}));
}

TEST_CASE(packed_affine_sphere_emitter_sample_and_pdf_are_consistent) {
    const CompiledScene scene = compile_scene(make_light_scene());
    const CompiledSceneView view = make_scene_view(scene);
    const std::uint32_t light_id =
        find_light(scene, PackedLightType::SphereEmitter);
    REQUIRE(light_id != kInvalidPackedIndex);
    const Float3 origin{0, 1, 4};
    bool found = false;
    for (int x = 1; x < 20 && !found; ++x) {
        for (int y = 1; y < 20 && !found; ++y) {
            PackedLightSample sample;
            const PackedLightStatus status = sample_packed_light(
                view, light_id, origin,
                {static_cast<float>(x) / 20.0f,
                 static_cast<float>(y) / 20.0f},
                sample);
            if (status != PackedLightStatus::Success) {
                continue;
            }
            float pdf = 0.0f;
            REQUIRE(evaluate_packed_light_pdf(view, light_id, origin,
                                              sample.wi, pdf) ==
                    PackedLightStatus::Success);
            REQUIRE_NEAR(pdf, sample.pdf, 2e-3);
            REQUIRE(sample.radiance.x > 0.0f);
            found = true;
        }
    }
    REQUIRE(found);
}

TEST_CASE(packed_mesh_emitter_sample_pdf_and_hit_mis_are_consistent) {
    const CompiledScene scene = compile_scene(make_light_scene());
    const CompiledSceneView view = make_scene_view(scene);
    const std::uint32_t light_id =
        find_light(scene, PackedLightType::MeshEmitter);
    REQUIRE(light_id != kInvalidPackedIndex);
    const Float3 origin{0, 0, 4};
    PackedLightSample sample;
    REQUIRE(sample_packed_light(view, light_id, origin, {0.4f, 0.7f},
                                sample) == PackedLightStatus::Success);
    float pdf = 0.0f;
    REQUIRE(evaluate_packed_light_pdf(view, light_id, origin, sample.wi,
                                      pdf) == PackedLightStatus::Success);
    REQUIRE_NEAR(pdf, sample.pdf, 2e-4);

    PackedHit hit;
    hit.t = sample.distance;
    hit.instance_id = scene.lights[light_id].instance_id;
    hit.primitive_id = sample.element_id;
    hit.flags = PACKED_HIT_TRIANGLE;
    float mis_pdf = 0.0f;
    REQUIRE(evaluate_packed_emitter_hit_mis_pdf(
                view, light_id, origin, sample.wi, hit, mis_pdf) ==
            PackedLightStatus::Success);
    REQUIRE_NEAR(mis_pdf,
                 sample.pdf * scene.lights[light_id].selection_probability,
                 2e-4);
}

TEST_CASE(packed_xz_rect_emitter_uses_the_cpu_compatible_light_normal) {
    SceneIR ir;
    ir.source_path = "packed_xz_emitter_test.json";
    ir.textures.push_back(
        {"emission", ConstantTextureIR{color(6.0, 5.0, 4.0)}});
    ir.materials.push_back(
        {"light", DiffuseLightMaterialIR{TextureIRId{0}}});
    AxisRectObjectIR rectangle;
    rectangle.plane = AxisRectPlane::XZ;
    rectangle.a0 = -1.0;
    rectangle.a1 = 1.0;
    rectangle.b0 = -1.0;
    rectangle.b1 = 1.0;
    rectangle.k = 2.0;
    rectangle.material = "light";
    ir.object_nodes.push_back({"ceiling light", rectangle});
    ir.objects = {0};
    ir.auto_emitters = true;

    const CompiledScene scene = compile_scene(ir);
    REQUIRE(validate_compiled_scene(scene).ok());
    const std::uint32_t light_id =
        find_light(scene, PackedLightType::MeshEmitter);
    REQUIRE(light_id != kInvalidPackedIndex);
    const PackedLight &light = scene.lights[light_id];
    REQUIRE(light.element_indices.count == 2);
    for (std::uint32_t local = 0; local < light.element_indices.count;
         ++local) {
        const std::uint32_t triangle_id = scene.light_element_indices[
            light.element_indices.offset + local];
        REQUIRE((scene.triangles[triangle_id].flags &
                 PACKED_TRIANGLE_REVERSE_EMITTER_NORMAL) != 0);
    }

    const CompiledSceneView view = make_scene_view(scene);
    PackedLightSample sample;
    REQUIRE(sample_packed_light(view, light_id, {0, 0, 0}, {0.3f, 0.7f},
                                sample) == PackedLightStatus::Success);
    REQUIRE(sample.wi.y > 0.0f);
    REQUIRE(sample.radiance.x > 0.0f);
    float pdf = 0.0f;
    REQUIRE(evaluate_packed_light_pdf(view, light_id, {0, 0, 0}, sample.wi,
                                      pdf) == PackedLightStatus::Success);
    REQUIRE_NEAR(pdf, sample.pdf, 2e-4);
}

TEST_CASE(packed_light_sampler_pdf_excludes_invisible_explicit_quad) {
    CompiledScene scene;
    scene.aggregates.push_back({});
    PackedLight visible;
    visible.type = PackedLightType::Quad;
    visible.flags = PACKED_LIGHT_BSDF_HITTABLE;
    visible.data0 = {-1, -1, 2, 1};
    visible.data1 = {2, 0, 0, 0};
    visible.data2 = {0, 2, 0, 0};
    PackedLight invisible = visible;
    invisible.flags = PACKED_LIGHT_NONE;
    scene.lights = {visible, invisible};
    scene.non_delta_light_indices = {0, 1};
    scene.light_selection_probabilities = {0.5f, 0.5f};
    scene.light_cdf = {0.5f, 1.0f};
    const CompiledSceneView view = make_scene_view(scene);
    const Float3 origin{0, 0, 4};
    const Float3 direction{0, 0, -1};
    float conditional = 0.0f;
    float mixture = 0.0f;
    REQUIRE(evaluate_packed_light_pdf(view, 0, origin, direction,
                                      conditional) ==
            PackedLightStatus::Success);
    REQUIRE(evaluate_packed_light_sampler_pdf(view, origin, direction,
                                              mixture) ==
            PackedLightStatus::Success);
    REQUIRE_NEAR(mixture, 0.5f * conditional, 1e-6);
}

TEST_CASE(packed_environment_sample_pdf_and_radiance_are_consistent) {
    const CompiledScene scene = make_constant_environment();
    REQUIRE(validate_compiled_scene(scene).ok());
    const CompiledSceneView view = make_scene_view(scene);
    PackedLightSample sample;
    REQUIRE(sample_packed_light(view, 0, {0, 0, 0}, {0.23f, 0.67f},
                                sample) == PackedLightStatus::Success);
    REQUIRE(sample.is_infinite());
    float pdf = 0.0f;
    REQUIRE(evaluate_packed_light_pdf(view, 0, {0, 0, 0}, sample.wi,
                                      pdf) == PackedLightStatus::Success);
    REQUIRE_NEAR(pdf, sample.pdf, 2e-5);
    Float3 radiance;
    REQUIRE(evaluate_packed_environment(view, 0, sample.wi, radiance) ==
            PackedLightStatus::Success);
    REQUIRE(nearly_equal(radiance, {1, 1, 1}, 1e-5f));
}

TEST_CASE(packed_non_delta_selection_matches_compiled_probabilities) {
    const CompiledScene scene = compile_scene(make_light_scene());
    const CompiledSceneView view = make_scene_view(scene);
    std::vector<int> counts(scene.non_delta_light_indices.size(), 0);
    RNG rng(12345);
    constexpr int kSamples = 40000;
    for (int index = 0; index < kSamples; ++index) {
        SelectedPackedLightSample sample;
        const PackedLightStatus status = sample_packed_non_delta_light(
            view, {0, 0, 5}, rng, sample);
        REQUIRE(status == PackedLightStatus::Success ||
                status == PackedLightStatus::NoSample);
        REQUIRE(sample.selection_index < counts.size());
        ++counts[sample.selection_index];
    }
    for (std::size_t index = 0; index < counts.size(); ++index) {
        const double frequency = static_cast<double>(counts[index]) / kSamples;
        REQUIRE_NEAR(frequency, scene.light_selection_probabilities[index],
                     0.015);
        REQUIRE(scene.light_selection_probabilities[index] >=
                0.05f / counts.size());
    }
}

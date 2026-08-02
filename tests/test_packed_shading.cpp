#include "test_harness.h"

#include "material_programs.h"
#include "packed_material.h"
#include "packed_texture.h"
#include "resource_compiler.h"

#include <memory>
#include <type_traits>

namespace {

PackedSurfaceInteraction test_surface() {
    PackedSurfaceInteraction surface;
    surface.position = {0.25f, -0.5f, 1.0f};
    surface.geometric_normal = {0.0f, 0.0f, 1.0f};
    surface.shading_normal = surface.geometric_normal;
    surface.dpdu = {1.0f, 0.0f, 0.0f};
    surface.dpdv = {0.0f, 1.0f, 0.0f};
    surface.uv = {0.35f, 0.65f};
    surface.vertex_color = {0.8f, 0.7f, 0.6f, 0.9f};
    surface.vertex_alpha = 0.9f;
    surface.flags = PACKED_HIT_FRONT_FACE;
    return surface;
}

TextureHandle constant_texture(const color &value) {
    return std::make_shared<SolidColorTexture>(value);
}

} // namespace

TEST_CASE(packed_shading_abi_is_fixed_and_trivially_copyable) {
    REQUIRE(sizeof(PackedSurfaceInteraction) == 112);
    REQUIRE(sizeof(PackedShadingFrame) == 32);
    REQUIRE(sizeof(PackedClosure) == 32);
    REQUIRE(sizeof(PackedMaterialOutput) == 320);
    REQUIRE(std::is_trivially_copyable_v<PackedSurfaceInteraction>);
    REQUIRE(std::is_trivially_copyable_v<PackedMaterialOutput>);
}

TEST_CASE(packed_texture_evaluator_reports_fixed_stack_overflow) {
    CompiledScene scene;
    PackedTextureNode constant;
    constant.type = PackedTextureType::Constant;
    constant.value0 = {0.25f, 0.5f, 0.75f, 1.0f};
    scene.texture_nodes.push_back(constant);
    for (std::uint32_t index = 1;
         index <= kPackedTextureStackCapacity; ++index) {
        PackedTextureNode scale;
        scale.type = PackedTextureType::Scale;
        scale.input0 = index - 1;
        scale.value0.x = 1.0f;
        scene.texture_nodes.push_back(scale);
    }

    Float4 sample{};
    REQUIRE(evaluate_packed_texture_status(
                make_scene_view(scene), kPackedTextureStackCapacity,
                PackedTextureEvalContext{}, sample) ==
            PackedShadingStatus::TextureStackOverflow);
}

TEST_CASE(packed_material_evaluator_emits_expected_closures) {
    CompiledScene scene;
    PackedResourceCompiler compiler(scene);
    const MaterialHandle lambertian_material =
        make_lambertian_material(color(0.2, 0.4, 0.6));
    const MaterialHandle mirror_material =
        make_metal_material(color(0.7, 0.8, 0.9), 0.0);
    const MaterialHandle metal_material =
        make_metal_material(color(0.6, 0.5, 0.4), 0.35);
    const MaterialHandle dielectric_material = make_dielectric_material(1.5);
    const MaterialHandle light_material =
        make_diffuse_light_material(color(3.0, 2.0, 1.0));
    const MaterialHandle phase_material =
        make_isotropic_material(color(0.3, 0.5, 0.7));
    const MaterialId lambertian =
        compiler.compile_material(lambertian_material);
    const MaterialId mirror = compiler.compile_material(mirror_material);
    const MaterialId metal = compiler.compile_material(metal_material);
    const MaterialId dielectric =
        compiler.compile_material(dielectric_material);
    const MaterialId light = compiler.compile_material(light_material);
    const MaterialId phase = compiler.compile_material(phase_material);
    const CompiledSceneView view = make_scene_view(scene);
    const PackedSurfaceInteraction surface = test_surface();
    PackedMaterialOutput output;

    REQUIRE(evaluate_packed_material(view, lambertian.value, surface,
                                     output));
    REQUIRE(output.closure_count == 1);
    REQUIRE(output.closures[0].type == PackedClosureType::Lambertian);
    REQUIRE_NEAR(output.closures[0].parameters.y, 0.4, 1e-6);

    REQUIRE(evaluate_packed_material(view, mirror.value, surface, output));
    REQUIRE(output.closure_count == 1);
    REQUIRE(output.closures[0].type == PackedClosureType::Mirror);

    REQUIRE(evaluate_packed_material(view, metal.value, surface, output));
    REQUIRE(output.closure_count == 1);
    REQUIRE(output.closures[0].type ==
            PackedClosureType::GGXReflection);
    REQUIRE_NEAR(output.closures[0].parameters.w, 0.35, 1e-6);

    REQUIRE(evaluate_packed_material(view, dielectric.value, surface,
                                     output));
    REQUIRE(output.closure_count == 1);
    REQUIRE(output.closures[0].type == PackedClosureType::Dielectric);
    REQUIRE((output.closures[0].flags & PACKED_CLOSURE_FRONT_FACE) != 0);
    REQUIRE_NEAR(output.closures[0].parameters.x, 1.5, 1e-6);

    REQUIRE(evaluate_packed_material(view, light.value, surface, output));
    REQUIRE(output.closure_count == 0);
    REQUIRE_NEAR(output.emission.x, 3.0, 1e-6);
    REQUIRE_NEAR(output.emission.y, 2.0, 1e-6);

    REQUIRE(evaluate_packed_material(view, phase.value, surface, output));
    REQUIRE(output.closure_count == 1);
    REQUIRE(output.closures[0].type == PackedClosureType::IsotropicPhase);
}

TEST_CASE(packed_principled_material_applies_normal_map_and_metadata) {
    const TextureHandle normal =
        constant_texture(color(0.5, 0.75, 1.0));
    const TextureHandle emission =
        constant_texture(color(0.25, 0.5, 1.0));
    const TextureHandle clearcoat = constant_texture(color(0.6, 0.6, 0.6));
    const TextureHandle coat_roughness =
        constant_texture(color(0.2, 0.2, 0.2));

    NormalMapSettings gl_settings;
    gl_settings.convention = NormalMapConvention::OpenGL;
    gl_settings.strength = 1.0;
    NormalMapSettings dx_settings = gl_settings;
    dx_settings.convention = NormalMapConvention::DirectX;

    CompiledScene scene;
    PackedResourceCompiler compiler(scene);
    const auto make_principled = [&](NormalMapSettings settings) {
        return make_principled_material(
            constant_texture(color(0.8, 0.4, 0.2)),
            constant_texture(color(0.3, 0.3, 0.3)),
            constant_texture(color(0.25, 0.25, 0.25)), normal, emission,
            2.0, clearcoat, coat_roughness, settings, true);
    };
    const MaterialHandle gl_material = make_principled(gl_settings);
    const MaterialHandle dx_material = make_principled(dx_settings);
    const MaterialId gl = compiler.compile_material(gl_material);
    const MaterialId dx = compiler.compile_material(dx_material);
    const CompiledSceneView view = make_scene_view(scene);
    const PackedSurfaceInteraction surface = test_surface();
    PackedMaterialOutput gl_output;
    PackedMaterialOutput dx_output;
    std::uint32_t gl_stack = 0;

    REQUIRE(evaluate_packed_material_status(
                view, gl.value, surface, gl_output, &gl_stack) ==
            PackedShadingStatus::Success);
    REQUIRE(evaluate_packed_material(view, dx.value, surface, dx_output));
    REQUIRE(gl_output.closure_count == 3);
    REQUIRE(gl_output.closures[0].type ==
            PackedClosureType::Lambertian);
    REQUIRE(gl_output.closures[1].type ==
            PackedClosureType::GGXReflection);
    REQUIRE(gl_output.closures[2].type ==
            PackedClosureType::ClearcoatGGX);
    REQUIRE_NEAR(gl_output.emission.x, 0.5, 1e-6);
    REQUIRE_NEAR(gl_output.emission.y, 1.0, 1e-6);
    REQUIRE_NEAR(gl_output.emission.z, 2.0, 1e-6);
    REQUIRE(gl_output.frame.normal.y > 0.4f);
    REQUIRE(dx_output.frame.normal.y < -0.4f);
    REQUIRE(gl_output.frame.normal.z > 0.8f);
    REQUIRE(dx_output.frame.normal.z > 0.8f);
    REQUIRE(gl_stack >= 1);
}

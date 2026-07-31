#include "transform.h"
#include "normal_mapping.h"
#include "resource_registry.h"
#include "test_harness.h"
#include "texture.h"

#include <filesystem>
#include <fstream>

namespace {

ShaderEvalContext context_at(double u, double v) {
    ShaderEvalContext context;
    context.uv0 = vec2(u, v);
    return context;
}

} // namespace

TEST_CASE(srgb_texture_uses_exact_transfer_function) {
    auto image = ImageAsset::from_pixels(
        1, 1, 3, {0.04045f, 0.5f, 1.0f});
    SamplerState sampler;
    sampler.wrap_u = WrapMode::Clamp;
    sampler.wrap_v = WrapMode::Clamp;
    sampler.filter = FilterMode::Nearest;
    ImageTexture texture(image, ColorSpace::SRGB, sampler);

    const color value = texture.evaluate(context_at(0.5, 0.5)).rgb;
    REQUIRE_NEAR(value.x(), 0.04045 / 12.92, 1e-7);
    REQUIRE_NEAR(value.y(), 0.214041140, 1e-6);
    REQUIRE_NEAR(value.z(), 1.0, 1e-12);
}

TEST_CASE(bilinear_filtering_happens_after_srgb_decode) {
    auto image =
        ImageAsset::from_pixels(2, 1, 3, {0, 0, 0, 1, 1, 1});
    SamplerState sampler;
    sampler.wrap_u = WrapMode::Clamp;
    sampler.wrap_v = WrapMode::Clamp;
    sampler.filter = FilterMode::Bilinear;
    ImageTexture texture(image, ColorSpace::SRGB, sampler);

    const color value = texture.evaluate(context_at(0.5, 0.5)).rgb;
    REQUIRE_NEAR(value.x(), 0.5, 1e-12);
    REQUIRE_NEAR(value.y(), 0.5, 1e-12);
    REQUIRE_NEAR(value.z(), 0.5, 1e-12);
}

TEST_CASE(image_texture_supports_wrap_and_channel_views) {
    auto image = ImageAsset::from_pixels(
        2, 1, 3, {0.1f, 0.2f, 0.3f, 0.7f, 0.8f, 0.9f});
    SamplerState sampler;
    sampler.filter = FilterMode::Nearest;
    ImageTexture repeated(image, ColorSpace::Linear, sampler,
                          TextureChannel::G);

    REQUIRE_NEAR(repeated.evaluate(context_at(1.75, 0.5)).rgb.x(), 0.8,
                 1e-6);

    sampler.wrap_u = WrapMode::Mirror;
    ImageTexture mirrored(image, ColorSpace::Linear, sampler,
                          TextureChannel::R);
    REQUIRE_NEAR(mirrored.evaluate(context_at(1.25, 0.5)).rgb.x(), 0.7,
                 1e-6);
}

TEST_CASE(image_assets_preserve_luminance_alpha_and_hdr_channels) {
    auto luminance = ImageAsset::from_pixels(1, 1, 1, {0.25f});
    REQUIRE_NEAR(luminance->component(0, 0, 0), 0.25, 1e-6);
    REQUIRE_NEAR(luminance->component(0, 0, 2), 0.25, 1e-6);
    REQUIRE_NEAR(luminance->component(0, 0, 3), 1.0, 1e-12);

    auto luminance_alpha =
        ImageAsset::from_pixels(1, 1, 2, {0.4f, 0.75f});
    REQUIRE_NEAR(luminance_alpha->component(0, 0, 1), 0.4, 1e-6);
    REQUIRE_NEAR(luminance_alpha->component(0, 0, 3), 0.75, 1e-6);

    auto rgba =
        ImageAsset::from_pixels(1, 1, 4, {0.1f, 0.2f, 0.3f, 0.4f});
    REQUIRE_NEAR(rgba->component(0, 0, 2), 0.3, 1e-6);
    REQUIRE_NEAR(rgba->component(0, 0, 3), 0.4, 1e-6);

    auto hdr = ImageAsset::from_pixels(1, 1, 3, {2.0f, 0.5f, 0.25f}, true);
    SamplerState sampler;
    sampler.wrap_u = WrapMode::Clamp;
    sampler.wrap_v = WrapMode::Clamp;
    sampler.filter = FilterMode::Nearest;
    ImageTexture hdr_view(hdr, ColorSpace::SRGB, sampler);
    const color value = hdr_view.evaluate(context_at(0.5, 0.5)).rgb;
    REQUIRE_NEAR(value.x(), 2.0, 1e-12);
    REQUIRE_NEAR(value.y(), 0.5, 1e-12);
}

TEST_CASE(resource_registry_deduplicates_and_caches_missing_images) {
    const auto path =
        std::filesystem::temp_directory_path() / "raytracer_texture_test.ppm";
    {
        std::ofstream output(path);
        output << "P3\n1 1\n255\n255 0 0\n";
    }

    ResourceRegistry resources;
    auto first = resources.load_image(path.string());
    auto second =
        resources.load_image((path.parent_path() / "." / path.filename())
                                 .string());
    REQUIRE(first == second);

    const std::string missing = path.string() + ".missing";
    auto missing_first = resources.load_image(missing);
    auto missing_second = resources.load_image(missing);
    REQUIRE(missing_first == ImageAsset::diagnostic());
    REQUIRE(missing_first == missing_second);
    REQUIRE(resources.image_count() == 2);
    std::filesystem::remove(path);
}

TEST_CASE(shading_frame_preserves_mirrored_uv_handedness) {
    ShadingFrame frame;
    frame.build_from_tangent_space(vec3(0, 0, 1), vec3(1, 0, 0),
                                   vec3(0, -1, 0));
    REQUIRE_NEAR(frame.handedness, -1.0, 1e-12);
    REQUIRE_NEAR(frame.bitangent.y(), -1.0, 1e-12);
}

TEST_CASE(normal_map_supports_opengl_directx_and_strength) {
    ShaderEvalContext context;
    context.geometry_normal = vec3(0, 0, 1);
    context.shading_normal = vec3(0, 0, 1);
    context.frame.build_from_normal(vec3(0, 0, 1));
    TextureHandle normal =
        std::make_shared<SolidColorTexture>(color(0.5, 1.0, 1.0));

    const ShadingFrame gl = apply_normal_map(context, normal);
    NormalMapSettings directx;
    directx.convention = NormalMapConvention::DirectX;
    const ShadingFrame dx = apply_normal_map(context, normal, directx);
    REQUIRE(gl.normal.y() > 0.6);
    REQUIRE(dx.normal.y() < -0.6);
    REQUIRE(gl.normal.z() > 0.6);
    REQUIRE(dx.normal.z() > 0.6);

    NormalMapSettings disabled;
    disabled.strength = 0.0;
    const ShadingFrame flat = apply_normal_map(context, normal, disabled);
    REQUIRE_NEAR(flat.normal.x(), 0.0, 1e-12);
    REQUIRE_NEAR(flat.normal.y(), 0.0, 1e-12);
    REQUIRE_NEAR(flat.normal.z(), 1.0, 1e-12);
}

TEST_CASE(normals_use_inverse_scale_transform) {
    const vec3 normal = unit_vector(vec3(1, 1, 0));
    const vec3 transformed = unit_vector(
        Transform::scale(vec3(2, 1, 1)).normal_to_world(normal));
    REQUIRE_NEAR(transformed.x(), 1.0 / std::sqrt(5.0), 1e-12);
    REQUIRE_NEAR(transformed.y(), 2.0 / std::sqrt(5.0), 1e-12);
}

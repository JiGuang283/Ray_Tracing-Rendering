#include "scene_resource_context.h"

#include "asset_path.h"
#include "material_programs.h"

#include <stdexcept>
#include <type_traits>

namespace scene_loader_internal {
namespace {

TextureHandle build_optional_texture(TextureIRId id,
                                     TextureSemantic semantic,
                                     SceneResourceContext &context) {
    return id == kInvalidTextureIR
               ? nullptr
               : build_texture(id, semantic, context);
}

TextureHandle compile_texture_node(const TextureIRData &data,
                                   TextureSemantic semantic,
                                   SceneResourceContext &context) {
    return std::visit(
        [&](const auto &typed) -> TextureHandle {
            using T = std::decay_t<decltype(typed)>;
            if constexpr (std::is_same_v<T, ConstantTextureIR>) {
                return make_shared<SolidColorTexture>(typed.value);
            } else if constexpr (std::is_same_v<T, AliasTextureIR>) {
                return build_texture(typed.target, semantic, context);
            } else if constexpr (std::is_same_v<T, CheckerTextureIR>) {
                return make_shared<CheckerTexture>(
                    build_texture(typed.even, semantic, context),
                    build_texture(typed.odd, semantic, context));
            } else if constexpr (std::is_same_v<T, NoiseTextureIR>) {
                return make_shared<NoiseTexture>(typed.scale);
            } else if constexpr (std::is_same_v<T, ImageTextureIR>) {
                ColorSpace color_space = ColorSpace::Linear;
                if (typed.color_space == TextureColorSpaceIR::SRGB ||
                    (typed.color_space == TextureColorSpaceIR::Auto &&
                     semantic == TextureSemantic::Color)) {
                    color_space = ColorSpace::SRGB;
                }
                TextureChannel channel = typed.channel;
                if (!typed.channel_explicit) {
                    channel = semantic == TextureSemantic::Scalar
                                  ? TextureChannel::R
                                  : TextureChannel::RGB;
                }
                const std::string path =
                    resolve_asset_path(context.source_path, typed.path);
                return make_shared<ImageTexture>(
                    context.resources.load_image(path), color_space,
                    typed.sampler, channel);
            } else if constexpr (std::is_same_v<T, ScaleTextureIR>) {
                return make_shared<ScaleTexture>(
                    build_texture(typed.input, semantic, context),
                    typed.scale);
            } else if constexpr (std::is_same_v<T, MultiplyTextureIR>) {
                return make_shared<MultiplyTexture>(
                    build_texture(typed.a, semantic, context),
                    build_texture(typed.b, semantic, context));
            } else if constexpr (std::is_same_v<T, MixTextureIR>) {
                return make_shared<MixTexture>(
                    build_texture(typed.a, semantic, context),
                    build_texture(typed.b, semantic, context),
                    build_texture(typed.factor, TextureSemantic::Scalar,
                                  context));
            } else if constexpr (std::is_same_v<T,
                                                ColorRampTextureIR>) {
                return make_shared<ColorRampTexture>(
                    build_texture(typed.input, TextureSemantic::Scalar,
                                  context),
                    typed.low, typed.high, typed.min_value,
                    typed.max_value);
            }
            throw std::runtime_error(
                "Scene file error: unsupported texture IR node.");
        },
        data);
}

} // namespace

TextureHandle build_texture(TextureIRId id, TextureSemantic semantic,
                            SceneResourceContext &context) {
    if (!context.scene_ir || id >= context.scene_ir->textures.size()) {
        throw std::runtime_error(
            "Scene file error: invalid texture IR id.");
    }

    const TextureCacheKey key{id, semantic};
    auto found = context.textures.find(key);
    if (found != context.textures.end()) {
        return found->second;
    }
    if (!context.textures_in_progress.insert(key).second) {
        throw std::runtime_error(
            "Scene file error: recursive texture compilation.");
    }

    try {
        TextureHandle texture = compile_texture_node(
            context.scene_ir->textures[id].data, semantic, context);
        context.textures.emplace(key, texture);
        context.textures_in_progress.erase(key);
        return texture;
    } catch (...) {
        context.textures_in_progress.erase(key);
        throw;
    }
}

MaterialHandle build_material(const MaterialIR &material,
                              SceneResourceContext &context) {
    return std::visit(
        [&](const auto &typed) -> MaterialHandle {
            using T = std::decay_t<decltype(typed)>;
            if constexpr (std::is_same_v<T, LambertianMaterialIR>) {
                return make_lambertian_material(build_texture(
                    typed.albedo, TextureSemantic::Color, context));
            } else if constexpr (std::is_same_v<T, MetalMaterialIR>) {
                return make_metal_material(typed.albedo,
                                           typed.roughness);
            } else if constexpr (std::is_same_v<T,
                                                DielectricMaterialIR>) {
                return make_dielectric_material(typed.ior);
            } else if constexpr (std::is_same_v<T,
                                                DiffuseLightMaterialIR>) {
                return make_diffuse_light_material(build_texture(
                    typed.emission, TextureSemantic::Color, context),
                    typed.double_sided);
            } else if constexpr (std::is_same_v<T,
                                                PrincipledMaterialIR>) {
                return make_principled_material(
                    build_texture(typed.base_color,
                                  TextureSemantic::Color, context),
                    build_texture(typed.roughness,
                                  TextureSemantic::Scalar, context),
                    build_texture(typed.metallic,
                                  TextureSemantic::Scalar, context),
                    build_optional_texture(typed.normal_map,
                                           TextureSemantic::Normal,
                                           context),
                    build_optional_texture(typed.emission,
                                           TextureSemantic::Color,
                                           context),
                    typed.emission_strength,
                    build_optional_texture(typed.clearcoat,
                                           TextureSemantic::Scalar,
                                           context),
                    build_optional_texture(typed.clearcoat_roughness,
                                           TextureSemantic::Scalar,
                                           context),
                    typed.normal_settings);
            }
            throw std::runtime_error(
                "Scene file error: unsupported material IR node.");
        },
        material.data);
}

MaterialHandle lookup_material(SceneResourceContext &context,
                               const std::string &name,
                               const std::string &context_name) {
    auto found = context.materials.find(name);
    if (found == context.materials.end()) {
        throw std::runtime_error("Scene file error: unknown material '" +
                                 name + "' in " + context_name + ".");
    }
    return found->second;
}

} // namespace scene_loader_internal

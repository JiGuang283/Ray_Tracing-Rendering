#include "resource_compiler.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace {

std::uint32_t checked_index(std::size_t value, const char *what) {
    if (value >= kInvalidPackedIndex) {
        throw std::overflow_error(std::string(what) +
                                  " exceeds the packed 32-bit limit");
    }
    return static_cast<std::uint32_t>(value);
}

Range32 append_range(std::size_t offset, std::size_t count,
                     const char *what) {
    if (offset >= kInvalidPackedIndex || count >= kInvalidPackedIndex ||
        count > kInvalidPackedIndex - offset) {
        throw std::overflow_error(std::string(what) +
                                  " exceeds the packed 32-bit limit");
    }
    return {static_cast<std::uint32_t>(offset),
            static_cast<std::uint32_t>(count)};
}

float checked_float(double value, const char *what) {
    if (!std::isfinite(value) ||
        value > static_cast<double>(std::numeric_limits<float>::max()) ||
        value < -static_cast<double>(std::numeric_limits<float>::max())) {
        throw std::runtime_error(std::string(what) +
                                 " is not representable as float32");
    }
    return static_cast<float>(value);
}

Float4 pack_color(const color &value, float w = 0.0f) {
    return {checked_float(value.x(), "color.x"),
            checked_float(value.y(), "color.y"),
            checked_float(value.z(), "color.z"), w};
}

std::uint32_t wrap_value(WrapMode mode) {
    switch (mode) {
    case WrapMode::Repeat:
        return 0;
    case WrapMode::Clamp:
        return 1;
    case WrapMode::Mirror:
        return 2;
    }
    return 0;
}

std::uint32_t sampler_flags(const ImageTexture &texture) {
    const SamplerState &sampler = texture.sampler();
    std::uint32_t flags =
        wrap_value(sampler.wrap_u) << PACKED_SAMPLER_WRAP_U_SHIFT;
    flags |= wrap_value(sampler.wrap_v) << PACKED_SAMPLER_WRAP_V_SHIFT;
    flags |= static_cast<std::uint32_t>(sampler.filter ==
                                        FilterMode::Bilinear)
             << PACKED_SAMPLER_FILTER_SHIFT;
    if (sampler.flip_v) {
        flags |= PACKED_SAMPLER_FLIP_V;
    }
    if (texture.color_space() == ColorSpace::SRGB) {
        flags |= PACKED_SAMPLER_SRGB;
    }
    return flags;
}

std::uint32_t texture_value(TextureId id) {
    return id.valid() ? id.value : kInvalidPackedIndex;
}

} // namespace

PackedResourceCompiler::PackedResourceCompiler(CompiledScene &scene)
    : m_scene(scene) {
}

ImageId PackedResourceCompiler::compile_image(
    const std::shared_ptr<const ImageAsset> &source_image) {
    const std::shared_ptr<const ImageAsset> image =
        source_image ? source_image : ImageAsset::diagnostic();
    auto found = m_images.find(image.get());
    if (found != m_images.end()) {
        return found->second;
    }

    const std::size_t texel_offset = m_scene.image_texels.size();
    const std::vector<float> &pixels = image->pixels();
    for (float value : pixels) {
        if (!std::isfinite(value)) {
            throw std::runtime_error(
                "image contains a non-finite texel value");
        }
    }
    m_scene.image_texels.insert(m_scene.image_texels.end(), pixels.begin(),
                                pixels.end());

    PackedImageDesc packed;
    packed.width = checked_index(static_cast<std::size_t>(image->width()),
                                 "image width");
    packed.height = checked_index(static_cast<std::size_t>(image->height()),
                                  "image height");
    packed.channels =
        checked_index(static_cast<std::size_t>(image->channels()),
                      "image channel count");
    packed.flags = image->is_hdr() ? PACKED_IMAGE_HDR : PACKED_IMAGE_NONE;
    packed.texels = append_range(texel_offset, pixels.size(), "image texels");

    const ImageId id{checked_index(m_scene.images.size(), "image count")};
    m_scene.images.push_back(packed);
    m_images.emplace(image.get(), id);
    return id;
}

PerlinId PackedResourceCompiler::compile_perlin(const perlin &noise) {
    auto found = m_perlin.find(&noise);
    if (found != m_perlin.end()) {
        return found->second;
    }

    PackedPerlinDesc packed;
    const auto &gradients = noise.gradients();
    const std::size_t gradient_offset = m_scene.perlin_gradients.size();
    for (const vec3 &gradient : gradients) {
        m_scene.perlin_gradients.push_back(pack_color(gradient));
    }
    packed.gradients =
        append_range(gradient_offset, gradients.size(), "Perlin gradients");

    auto append_permutation = [&](const std::vector<int> &source) {
        const std::size_t offset = m_scene.perlin_permutations.size();
        for (int value : source) {
            if (value < 0) {
                throw std::runtime_error(
                    "Perlin permutation contains a negative index");
            }
            m_scene.perlin_permutations.push_back(
                static_cast<std::uint32_t>(value));
        }
        return append_range(offset, source.size(), "Perlin permutation");
    };
    packed.permutation_x = append_permutation(noise.permutation_x());
    packed.permutation_y = append_permutation(noise.permutation_y());
    packed.permutation_z = append_permutation(noise.permutation_z());

    const PerlinId id{
        checked_index(m_scene.perlin_tables.size(), "Perlin table count")};
    m_scene.perlin_tables.push_back(packed);
    m_perlin.emplace(&noise, id);
    return id;
}

TextureId
PackedResourceCompiler::compile_texture(const TextureHandle &texture) {
    if (!texture) {
        return {};
    }
    auto found = m_textures.find(texture.get());
    if (found != m_textures.end()) {
        return found->second;
    }
    if (!m_textures_in_progress.insert(texture.get()).second) {
        throw std::runtime_error("texture graph contains a cycle");
    }

    PackedTextureNode node;
    switch (texture->kind()) {
    case TextureKind::SolidColor: {
        const auto &typed = static_cast<const SolidColorTexture &>(*texture);
        node.type = PackedTextureType::Constant;
        node.value0 = pack_color(typed.value(), 1.0f);
        break;
    }
    case TextureKind::VertexColor:
        node.type = PackedTextureType::VertexColor;
        break;
    case TextureKind::Checker: {
        const auto &typed = static_cast<const CheckerTexture &>(*texture);
        node.type = PackedTextureType::Checker;
        node.input0 = texture_value(compile_texture(typed.even()));
        node.input1 = texture_value(compile_texture(typed.odd()));
        break;
    }
    case TextureKind::Image: {
        const auto &typed = static_cast<const ImageTexture &>(*texture);
        node.type = PackedTextureType::Image;
        node.image_id = compile_image(typed.image()).value;
        node.sampler_flags = sampler_flags(typed);
        node.channel = static_cast<std::uint32_t>(typed.channel());
        break;
    }
    case TextureKind::Noise: {
        const auto &typed = static_cast<const NoiseTexture &>(*texture);
        node.type = PackedTextureType::Noise;
        node.perlin_id = compile_perlin(typed.noise_data()).value;
        node.value0.x = checked_float(typed.scale(), "noise scale");
        break;
    }
    case TextureKind::Scale: {
        const auto &typed = static_cast<const ScaleTexture &>(*texture);
        node.type = PackedTextureType::Scale;
        node.input0 = texture_value(compile_texture(typed.input()));
        node.value0.x = checked_float(typed.scale(), "texture scale");
        break;
    }
    case TextureKind::UVTransform: {
        const auto &typed = static_cast<const UVTransformTexture &>(*texture);
        node.type = PackedTextureType::UVTransform;
        node.input0 = texture_value(compile_texture(typed.input()));
        node.value0 = {checked_float(typed.offset().x(), "UV offset.x"),
                       checked_float(typed.offset().y(), "UV offset.y"),
                       checked_float(typed.scale().x(), "UV scale.x"),
                       checked_float(typed.scale().y(), "UV scale.y")};
        node.value1.x =
            checked_float(typed.cos_rotation(), "UV rotation cosine");
        node.value1.y =
            checked_float(typed.sin_rotation(), "UV rotation sine");
        break;
    }
    case TextureKind::Multiply: {
        const auto &typed = static_cast<const MultiplyTexture &>(*texture);
        node.type = PackedTextureType::Multiply;
        node.input0 = texture_value(compile_texture(typed.a()));
        node.input1 = texture_value(compile_texture(typed.b()));
        break;
    }
    case TextureKind::Mix: {
        const auto &typed = static_cast<const MixTexture &>(*texture);
        node.type = PackedTextureType::Mix;
        node.input0 = texture_value(compile_texture(typed.a()));
        node.input1 = texture_value(compile_texture(typed.b()));
        node.input2 = texture_value(compile_texture(typed.factor()));
        break;
    }
    case TextureKind::ColorRamp: {
        const auto &typed = static_cast<const ColorRampTexture &>(*texture);
        node.type = PackedTextureType::ColorRamp;
        node.input0 = texture_value(compile_texture(typed.input()));
        node.value0 = pack_color(typed.low());
        node.value1 = pack_color(typed.high());
        node.value2.x = checked_float(typed.min_value(), "ramp minimum");
        node.value2.y = checked_float(typed.max_value(), "ramp maximum");
        break;
    }
    case TextureKind::Unsupported:
        m_textures_in_progress.erase(texture.get());
        throw std::runtime_error(
            "material uses a texture type that cannot be packed");
    }

    const TextureId id{
        checked_index(m_scene.texture_nodes.size(), "texture node count")};
    m_scene.texture_nodes.push_back(node);
    m_textures.emplace(texture.get(), id);
    m_textures_in_progress.erase(texture.get());
    return id;
}

MaterialId
PackedResourceCompiler::compile_material(const MaterialHandle &material) {
    if (!material) {
        throw std::runtime_error("cannot pack a null material");
    }
    auto found = m_materials.find(material.get());
    if (found != m_materials.end()) {
        return found->second;
    }

    PackedMaterial packed;
    if (material->is_emissive()) {
        packed.flags |= PACKED_MATERIAL_EMISSIVE;
    }
    if (material->is_double_sided()) {
        packed.flags |= PACKED_MATERIAL_DOUBLE_SIDED;
    }
    packed.emission_estimate = pack_color(material->emission_estimate());

    auto set_texture = [&](std::size_t slot, const TextureHandle &texture) {
        packed.texture_ids[slot] = texture_value(compile_texture(texture));
    };

    std::visit(
        [&](const auto &description) {
            using T = std::decay_t<decltype(description)>;
            if constexpr (std::is_same_v<T, std::monostate>) {
                throw std::runtime_error(
                    "material has no host compile description");
            } else if constexpr (std::is_same_v<
                                     T, LambertianMaterialDescription>) {
                packed.type = PackedMaterialType::Lambertian;
                set_texture(0, description.albedo);
            } else if constexpr (std::is_same_v<
                                     T, MetalMaterialDescription>) {
                packed.type = PackedMaterialType::Metal;
                packed.parameters[0] = pack_color(
                    description.albedo,
                    checked_float(description.roughness,
                                  "metal roughness"));
            } else if constexpr (std::is_same_v<
                                     T, DielectricMaterialDescription>) {
                packed.type = PackedMaterialType::Dielectric;
                packed.parameters[0].x =
                    checked_float(description.ior, "dielectric IOR");
            } else if constexpr (std::is_same_v<
                                     T, DiffuseLightMaterialDescription>) {
                packed.type = PackedMaterialType::DiffuseLight;
                set_texture(0, description.emission);
            } else if constexpr (std::is_same_v<
                                     T, PrincipledMaterialDescription>) {
                packed.type = PackedMaterialType::Principled;
                set_texture(0, description.base_color);
                set_texture(1, description.roughness);
                set_texture(2, description.metallic);
                set_texture(3, description.normal_map);
                set_texture(4, description.emission);
                set_texture(5, description.clearcoat);
                set_texture(6, description.clearcoat_roughness);
                packed.parameters[0].x = checked_float(
                    description.emission_strength, "emission strength");
                packed.parameters[0].y = checked_float(
                    description.normal_strength, "normal strength");
                if (description.normal_convention != 0) {
                    packed.flags |= PACKED_MATERIAL_NORMAL_DIRECTX;
                }
            } else if constexpr (std::is_same_v<
                                     T, IsotropicMaterialDescription>) {
                packed.type = PackedMaterialType::Isotropic;
                set_texture(0, description.albedo);
            }
        },
        material->description());

    const MaterialId id{
        checked_index(m_scene.materials.size(), "material count")};
    m_scene.materials.push_back(packed);
    m_materials.emplace(material.get(), id);
    return id;
}

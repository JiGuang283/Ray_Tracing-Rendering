#include "scene_loader_internal.h"

#include "normal_mapping.h"

#include <stdexcept>

namespace scene_loader_internal {
namespace {

const json &material_field_or(const json &material_json,
                              const std::string &first_name,
                              const std::string &second_name,
                              const std::string &context) {
    auto first = material_json.find(first_name);
    return first != material_json.end()
               ? *first
               : require_key(material_json, second_name, context);
}

ColorSpace parse_color_space(const json &texture_json,
                             const std::string &context) {
    const std::string value =
        texture_json.value("color_space", std::string("srgb"));
    if (value == "srgb") {
        return ColorSpace::SRGB;
    }
    if (value == "linear") {
        return ColorSpace::Linear;
    }
    throw std::runtime_error("Scene file error: unknown color_space '" +
                             value + "' in " + context + ".");
}

WrapMode parse_wrap_mode(const std::string &value,
                         const std::string &context) {
    if (value == "repeat") {
        return WrapMode::Repeat;
    }
    if (value == "clamp") {
        return WrapMode::Clamp;
    }
    if (value == "mirror") {
        return WrapMode::Mirror;
    }
    throw std::runtime_error("Scene file error: unknown wrap mode '" + value +
                             "' in " + context + ".");
}

FilterMode parse_filter_mode(const json &texture_json,
                             const std::string &context) {
    const std::string value =
        texture_json.value("filter", std::string("bilinear"));
    if (value == "nearest") {
        return FilterMode::Nearest;
    }
    if (value == "bilinear") {
        return FilterMode::Bilinear;
    }
    throw std::runtime_error("Scene file error: unknown filter mode '" +
                             value + "' in " + context + ".");
}

TextureChannel parse_texture_channel(const json &texture_json,
                                     const std::string &context) {
    const std::string value =
        texture_json.value("channel", std::string("rgb"));
    if (value == "rgb") {
        return TextureChannel::RGB;
    }
    if (value == "r") {
        return TextureChannel::R;
    }
    if (value == "g") {
        return TextureChannel::G;
    }
    if (value == "b") {
        return TextureChannel::B;
    }
    if (value == "a") {
        return TextureChannel::A;
    }
    throw std::runtime_error("Scene file error: unknown texture channel '" +
                             value + "' in " + context + ".");
}

NormalMapConvention parse_normal_convention(const std::string &value,
                                            const std::string &context) {
    if (value == "opengl") {
        return NormalMapConvention::OpenGL;
    }
    if (value == "directx") {
        return NormalMapConvention::DirectX;
    }
    throw std::runtime_error(
        "Scene file error: unknown normal map convention '" + value +
        "' in " + context + ".");
}

struct NormalInput {
    TextureHandle texture;
    NormalMapSettings settings;
};

NormalInput build_normal_input(const json &material_json,
                               SceneBuildContext &context,
                               const std::string &name) {
    NormalInput input;
    const std::string field_context =
        "material '" + name + "'.normal_map";
    if (material_json.contains("normal_map")) {
        const json &normal_json = material_json["normal_map"];
        if (!normal_json.is_object() || !normal_json.contains("texture")) {
            throw std::runtime_error(
                "Scene file error: normal_map requires a texture in " +
                field_context + ".");
        }
        input.texture =
            build_texture_value(normal_json["texture"], context,
                                field_context + ".texture");
        input.settings.strength =
            read_double_or(normal_json, "strength", 1.0);
        input.settings.convention = parse_normal_convention(
            normal_json.value("convention", std::string("opengl")),
            field_context);
    } else if (material_json.contains("normal")) {
        input.texture =
            build_texture_value(material_json["normal"], context,
                                "material '" + name + "'.normal");
    } else if (material_json.contains("normal_texture")) {
        input.texture = build_texture_value(
            material_json["normal_texture"], context,
            "material '" + name + "'.normal_texture");
    }
    return input;
}

} // namespace

TextureHandle lookup_texture(SceneBuildContext &context,
                             const std::string &name,
                             const std::string &context_name) {
    auto built = context.textures.find(name);
    if (built != context.textures.end()) {
        if (!built->second) {
            throw std::runtime_error("Scene file error: recursive texture '" +
                                     name + "' in " + context_name + ".");
        }
        return built->second;
    }

    auto found = context.texture_specs.find(name);
    if (found == context.texture_specs.end()) {
        throw std::runtime_error("Scene file error: unknown texture '" + name +
                                 "' in " + context_name + ".");
    }
    context.textures[name] = nullptr;
    context.textures[name] =
        build_texture_value(found->second, context, "texture '" + name + "'");
    return context.textures[name];
}

TextureHandle build_texture_value(const json &texture_json,
                                  SceneBuildContext &context,
                                  const std::string &context_name);

TextureHandle build_texture_object(const json &texture_json,
                                   SceneBuildContext &context,
                                   const std::string &context_name) {
    if (texture_json.contains("ref")) {
        return lookup_texture(context, texture_json["ref"].get<std::string>(),
                              context_name);
    }

    const std::string type =
        read_string(texture_json, "type", context_name);
    if (type == "solid") {
        if (texture_json.contains("color")) {
            return make_shared<SolidColorTexture>(
                read_vec3(texture_json, "color", context_name));
        }
        if (texture_json.contains("value")) {
            const auto &value = texture_json["value"];
            if (value.is_number()) {
                const double scalar = value.get<double>();
                return make_shared<SolidColorTexture>(scalar, scalar, scalar);
            }
            return make_shared<SolidColorTexture>(
                read_vec3_value(value, context_name + ".value"));
        }
        throw std::runtime_error(
            "Scene file error: solid texture needs 'color' or 'value' in " +
            context_name + ".");
    }
    if (type == "checker") {
        const json &even =
            texture_json.contains("even")
                ? texture_json["even"]
                : require_key(texture_json, "color1", context_name);
        const json &odd =
            texture_json.contains("odd")
                ? texture_json["odd"]
                : require_key(texture_json, "color2", context_name);
        return make_shared<CheckerTexture>(
            build_texture_value(even, context, context_name + ".even"),
            build_texture_value(odd, context, context_name + ".odd"));
    }
    if (type == "noise") {
        return make_shared<NoiseTexture>(
            read_double_or(texture_json, "scale", 1.0));
    }
    if (type == "image") {
        SamplerState sampler;
        sampler.wrap_u = parse_wrap_mode(
            texture_json.value("wrap_u", std::string("repeat")),
            context_name + ".wrap_u");
        sampler.wrap_v = parse_wrap_mode(
            texture_json.value("wrap_v", std::string("repeat")),
            context_name + ".wrap_v");
        sampler.filter = parse_filter_mode(texture_json, context_name);

        const std::string path = resolve_asset_path(
            context, read_string(texture_json, "path", context_name));
        return make_shared<ImageTexture>(
            context.resources.load_image(path),
            parse_color_space(texture_json, context_name), sampler,
            parse_texture_channel(texture_json, context_name));
    }
    if (type == "scale") {
        return make_shared<ScaleTexture>(
            build_texture_value(
                require_key(texture_json, "input", context_name), context,
                context_name + ".input"),
            read_double_or(texture_json, "scale", 1.0));
    }
    if (type == "multiply") {
        return make_shared<MultiplyTexture>(
            build_texture_value(require_key(texture_json, "a", context_name),
                                context, context_name + ".a"),
            build_texture_value(require_key(texture_json, "b", context_name),
                                context, context_name + ".b"));
    }
    if (type == "mix") {
        return make_shared<MixTexture>(
            build_texture_value(require_key(texture_json, "a", context_name),
                                context, context_name + ".a"),
            build_texture_value(require_key(texture_json, "b", context_name),
                                context, context_name + ".b"),
            build_texture_value(
                texture_json.contains("factor") ? texture_json["factor"]
                                                : json(0.5),
                context, context_name + ".factor"));
    }
    if (type == "color_ramp") {
        return make_shared<ColorRampTexture>(
            build_texture_value(
                require_key(texture_json, "input", context_name), context,
                context_name + ".input"),
            read_vec3(texture_json, "low", context_name),
            read_vec3(texture_json, "high", context_name),
            read_double_or(texture_json, "min", 0.0),
            read_double_or(texture_json, "max", 1.0));
    }
    throw std::runtime_error("Scene file error: unknown texture type '" + type +
                             "' in " + context_name + ".");
}

TextureHandle build_texture_value(const json &texture_json,
                                  SceneBuildContext &context,
                                  const std::string &context_name) {
    if (texture_json.is_string()) {
        return lookup_texture(context, texture_json.get<std::string>(),
                              context_name);
    }
    if (texture_json.is_number()) {
        const double scalar = texture_json.get<double>();
        return make_shared<SolidColorTexture>(scalar, scalar, scalar);
    }
    if (texture_json.is_array()) {
        return make_shared<SolidColorTexture>(
            read_vec3_value(texture_json, context_name));
    }
    if (texture_json.is_object()) {
        return build_texture_object(texture_json, context, context_name);
    }
    throw std::runtime_error("Scene file error: invalid texture value in " +
                             context_name + ".");
}

shared_ptr<material>
build_material(const json &material_json, SceneBuildContext &context,
               const std::string &name) {
    const std::string type =
        read_string(material_json, "type", "material '" + name + "'");

    if (type == "lambertian") {
        const auto &albedo =
            material_json.contains("texture")
                ? material_json["texture"]
                : material_field_or(material_json, "albedo", "color",
                                    "material '" + name + "'");
        return make_shared<lambertian>(
            build_texture_value(albedo, context, "material '" + name + "'"));
    }
    if (type == "metal") {
        return make_shared<metal>(
            read_vec3(material_json, "albedo", "material '" + name + "'"),
            read_double_or(material_json, "fuzz", 0.0));
    }
    if (type == "dielectric") {
        const double ior =
            read_double_or(material_json, "ir",
                           read_double_or(material_json,
                                          "index_of_refraction", 1.5));
        return make_shared<dielectric>(ior);
    }
    if (type == "diffuse_light") {
        const auto &emission =
            material_json.contains("texture")
                ? material_json["texture"]
                : material_field_or(material_json, "emit", "color",
                                    "material '" + name + "'");
        return make_shared<diffuse_light>(build_texture_value(
            emission, context, "material '" + name + "'"));
    }
    if (type == "pbr" || type == "principled") {
        const json &base_color =
            material_json.contains("base_color")
                ? material_json["base_color"]
                : require_key(material_json, "albedo",
                              "material '" + name + "'");
        const NormalInput normal =
            build_normal_input(material_json, context, name);
        return make_shared<PrincipledMaterial>(
            build_texture_value(base_color, context,
                                "material '" + name + "'.base_color"),
            build_texture_value(
                material_json.contains("roughness")
                    ? material_json["roughness"]
                    : json(0.5),
                context, "material '" + name + "'.roughness"),
            build_texture_value(
                material_json.contains("metallic")
                    ? material_json["metallic"]
                    : json(0.0),
                context, "material '" + name + "'.metallic"),
            normal.texture,
            material_json.contains("emission")
                ? build_texture_value(material_json["emission"], context,
                                      "material '" + name + "'.emission")
                : nullptr,
            read_double_or(material_json, "emission_strength", 1.0),
            material_json.contains("clearcoat")
                ? build_texture_value(material_json["clearcoat"], context,
                                      "material '" + name + "'.clearcoat")
                : nullptr,
            material_json.contains("clearcoat_roughness")
                ? build_texture_value(
                      material_json["clearcoat_roughness"], context,
                      "material '" + name + "'.clearcoat_roughness")
                : nullptr,
            normal.settings);
    }
    throw std::runtime_error("Scene file error: unknown material type '" + type +
                             "' for material '" + name + "'.");
}

shared_ptr<material>
lookup_material(SceneBuildContext &context, const json &object,
                const std::string &context_name) {
    const std::string material_name =
        read_string(object, "material", context_name);
    auto built = context.materials.find(material_name);
    if (built != context.materials.end()) {
        if (!built->second) {
            throw std::runtime_error("Scene file error: recursive material '" +
                                     material_name + "' in " + context_name +
                                     ".");
        }
        return built->second;
    }

    auto found = context.material_specs.find(material_name);
    if (found == context.material_specs.end()) {
        throw std::runtime_error("Scene file error: unknown material '" +
                                 material_name + "' in " + context_name +
                                 ".");
    }
    context.materials[material_name] = nullptr;
    context.materials[material_name] =
        build_material(found->second, context, material_name);
    return context.materials[material_name];
}

} // namespace scene_loader_internal

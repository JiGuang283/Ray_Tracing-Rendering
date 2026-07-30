#include "scene_loader_internal.h"

#include <stdexcept>

namespace scene_loader_internal {

const json &material_field_or(const json &material_json, const std::string &a,
                              const std::string &b,
                              const std::string &context) {
    auto first = material_json.find(a);
    if (first != material_json.end()) {
        return *first;
    }
    return require_key(material_json, b, context);
}

shared_ptr<texture> lookup_texture(SceneBuildContext &context,
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

shared_ptr<texture> build_texture_value(const json &texture_json,
                                        SceneBuildContext &context,
                                        const std::string &context_name);

shared_ptr<texture> build_texture_object(const json &texture_json,
                                         SceneBuildContext &context,
                                         const std::string &context_name) {
    if (texture_json.contains("ref")) {
        return lookup_texture(context, texture_json["ref"].get<std::string>(),
                              context_name);
    }

    std::string type = read_string(texture_json, "type", context_name);
    if (type == "solid") {
        if (texture_json.contains("color")) {
            return make_shared<solid_color>(
                read_vec3(texture_json, "color", context_name));
        }
        if (texture_json.contains("value")) {
            const auto &value = texture_json["value"];
            if (value.is_number()) {
                double scalar = value.get<double>();
                return make_shared<solid_color>(scalar, scalar, scalar);
            }
            return make_shared<solid_color>(
                read_vec3_value(value, context_name + ".value"));
        }
        throw std::runtime_error(
            "Scene file error: solid texture needs 'color' or 'value' in " +
            context_name + ".");
    }
    if (type == "checker") {
        const json &even =
            texture_json.contains("even") ? texture_json["even"]
                                          : require_key(texture_json, "color1",
                                                        context_name);
        const json &odd =
            texture_json.contains("odd") ? texture_json["odd"]
                                         : require_key(texture_json, "color2",
                                                       context_name);
        return make_shared<checker_texture>(
            build_texture_value(even, context, context_name + ".even"),
            build_texture_value(odd, context, context_name + ".odd"));
    }
    if (type == "noise") {
        return make_shared<noise_texture>(
            read_double_or(texture_json, "scale", 1.0));
    }
    if (type == "image") {
        return make_shared<image_texture>(
            resolve_asset_path(context,
                               read_string(texture_json, "path",
                                           context_name))
                .c_str());
    }
    if (type == "scale") {
        return make_shared<scale_texture>(
            build_texture_value(require_key(texture_json, "input",
                                            context_name),
                                context, context_name + ".input"),
            read_double_or(texture_json, "scale", 1.0));
    }
    if (type == "multiply") {
        return make_shared<multiply_texture>(
            build_texture_value(require_key(texture_json, "a", context_name),
                                context, context_name + ".a"),
            build_texture_value(require_key(texture_json, "b", context_name),
                                context, context_name + ".b"));
    }
    if (type == "mix") {
        return make_shared<mix_texture>(
            build_texture_value(require_key(texture_json, "a", context_name),
                                context, context_name + ".a"),
            build_texture_value(require_key(texture_json, "b", context_name),
                                context, context_name + ".b"),
            build_texture_value(texture_json.contains("factor")
                                    ? texture_json["factor"]
                                    : json(0.5),
                                context, context_name + ".factor"));
    }
    if (type == "color_ramp") {
        return make_shared<color_ramp_texture>(
            build_texture_value(require_key(texture_json, "input",
                                            context_name),
                                context, context_name + ".input"),
            read_vec3(texture_json, "low", context_name),
            read_vec3(texture_json, "high", context_name),
            read_double_or(texture_json, "min", 0.0),
            read_double_or(texture_json, "max", 1.0));
    }

    throw std::runtime_error("Scene file error: unknown texture type '" + type +
                             "' in " + context_name + ".");
}

shared_ptr<texture> build_texture_value(const json &texture_json,
                                        SceneBuildContext &context,
                                        const std::string &context_name) {
    if (texture_json.is_string()) {
        return lookup_texture(context, texture_json.get<std::string>(),
                              context_name);
    }
    if (texture_json.is_number()) {
        double scalar = texture_json.get<double>();
        return make_shared<solid_color>(scalar, scalar, scalar);
    }
    if (texture_json.is_array()) {
        return make_shared<solid_color>(
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
    std::string type = read_string(material_json, "type", "material '" + name +
                                                        "'");

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
        double ir = read_double_or(
            material_json, "ir",
            read_double_or(material_json, "index_of_refraction", 1.5));
        return make_shared<dielectric>(ir);
    }
    if (type == "diffuse_light") {
        const auto &emit =
            material_json.contains("texture")
                ? material_json["texture"]
                : material_field_or(material_json, "emit", "color",
                                    "material '" + name + "'");
        return make_shared<diffuse_light>(
            build_texture_value(emit, context, "material '" + name + "'"));
    }
    if (type == "pbr" || type == "principled") {
        const json &base_color =
            material_json.contains("base_color")
                ? material_json["base_color"]
                : require_key(material_json, "albedo",
                              "material '" + name + "'");
        return make_shared<PrincipledMaterial>(
            build_texture_value(base_color, context,
                                "material '" + name + "'.base_color"),
            build_texture_value(material_json.contains("roughness")
                                    ? material_json["roughness"]
                                    : json(0.5),
                                context, "material '" + name + "'.roughness"),
            build_texture_value(material_json.contains("metallic")
                                    ? material_json["metallic"]
                                    : json(0.0),
                                context, "material '" + name + "'.metallic"),
            material_json.contains("normal")
                ? build_texture_value(material_json["normal"], context,
                                      "material '" + name + "'.normal")
                : nullptr,
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
                : nullptr);
    }

    throw std::runtime_error("Scene file error: unknown material type '" + type +
                             "' for material '" + name + "'.");
}

shared_ptr<material>
lookup_material(SceneBuildContext &build_context, const json &object,
                const std::string &context_name) {
    std::string material_name =
        read_string(object, "material", context_name);
    auto built = build_context.materials.find(material_name);
    if (built != build_context.materials.end()) {
        if (!built->second) {
            throw std::runtime_error("Scene file error: recursive material '" +
                                     material_name + "' in " + context_name +
                                     ".");
        }
        return built->second;
    }

    auto found = build_context.material_specs.find(material_name);
    if (found == build_context.material_specs.end()) {
        throw std::runtime_error("Scene file error: unknown material '" +
                                 material_name + "' in " + context_name + ".");
    }
    build_context.materials[material_name] = nullptr;
    build_context.materials[material_name] =
        build_material(found->second, build_context, material_name);
    return build_context.materials[material_name];
}


} // namespace scene_loader_internal

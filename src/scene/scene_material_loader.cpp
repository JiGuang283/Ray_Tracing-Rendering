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

shared_ptr<texture> lookup_texture(const TextureMap &textures,
                                   const std::string &name,
                                   const std::string &context) {
    auto found = textures.find(name);
    if (found == textures.end()) {
        throw std::runtime_error("Scene file error: unknown texture '" + name +
                                 "' in " + context + ".");
    }
    return found->second;
}

shared_ptr<texture> build_texture_value(const json &texture_json,
                                        const TextureMap &textures,
                                        const std::string &context);

shared_ptr<texture> build_texture_object(const json &texture_json,
                                         const TextureMap &textures,
                                         const std::string &context) {
    if (texture_json.contains("ref")) {
        return lookup_texture(textures, texture_json["ref"].get<std::string>(),
                              context);
    }

    std::string type = read_string(texture_json, "type", context);
    if (type == "solid") {
        if (texture_json.contains("color")) {
            return make_shared<solid_color>(
                read_vec3(texture_json, "color", context));
        }
        if (texture_json.contains("value")) {
            const auto &value = texture_json["value"];
            if (value.is_number()) {
                double scalar = value.get<double>();
                return make_shared<solid_color>(scalar, scalar, scalar);
            }
            return make_shared<solid_color>(
                read_vec3_value(value, context + ".value"));
        }
        throw std::runtime_error(
            "Scene file error: solid texture needs 'color' or 'value' in " +
            context + ".");
    }
    if (type == "checker") {
        const json &even =
            texture_json.contains("even") ? texture_json["even"]
                                          : require_key(texture_json, "color1",
                                                        context);
        const json &odd =
            texture_json.contains("odd") ? texture_json["odd"]
                                         : require_key(texture_json, "color2",
                                                       context);
        return make_shared<checker_texture>(
            build_texture_value(even, textures, context + ".even"),
            build_texture_value(odd, textures, context + ".odd"));
    }
    if (type == "noise") {
        return make_shared<noise_texture>(
            read_double_or(texture_json, "scale", 1.0));
    }
    if (type == "image") {
        return make_shared<image_texture>(
            read_string(texture_json, "path", context).c_str());
    }

    throw std::runtime_error("Scene file error: unknown texture type '" + type +
                             "' in " + context + ".");
}

shared_ptr<texture> build_texture_value(const json &texture_json,
                                        const TextureMap &textures,
                                        const std::string &context) {
    if (texture_json.is_string()) {
        return lookup_texture(textures, texture_json.get<std::string>(),
                              context);
    }
    if (texture_json.is_number()) {
        double scalar = texture_json.get<double>();
        return make_shared<solid_color>(scalar, scalar, scalar);
    }
    if (texture_json.is_array()) {
        return make_shared<solid_color>(
            read_vec3_value(texture_json, context));
    }
    if (texture_json.is_object()) {
        return build_texture_object(texture_json, textures, context);
    }
    throw std::runtime_error("Scene file error: invalid texture value in " +
                             context + ".");
}

shared_ptr<material>
build_material(const json &material_json, const TextureMap &textures,
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
            build_texture_value(albedo, textures, "material '" + name + "'"));
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
            build_texture_value(emit, textures, "material '" + name + "'"));
    }
    if (type == "pbr") {
        return make_shared<PBRMaterial>(
            build_texture_value(require_key(material_json, "albedo",
                                            "material '" + name + "'"),
                                textures, "material '" + name + "'.albedo"),
            build_texture_value(material_json.contains("roughness")
                                    ? material_json["roughness"]
                                    : json(0.5),
                                textures, "material '" + name + "'.roughness"),
            build_texture_value(material_json.contains("metallic")
                                    ? material_json["metallic"]
                                    : json(0.0),
                                textures, "material '" + name + "'.metallic"),
            material_json.contains("normal")
                ? build_texture_value(material_json["normal"], textures,
                                      "material '" + name + "'.normal")
                : nullptr);
    }

    throw std::runtime_error("Scene file error: unknown material type '" + type +
                             "' for material '" + name + "'.");
}

shared_ptr<material>
lookup_material(const MaterialMap &materials, const json &object,
                const std::string &context) {
    std::string material_name = read_string(object, "material", context);
    auto found = materials.find(material_name);
    if (found == materials.end()) {
        throw std::runtime_error("Scene file error: unknown material '" +
                                 material_name + "' in " + context + ".");
    }
    return found->second;
}


} // namespace scene_loader_internal

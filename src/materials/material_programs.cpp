#include "material_programs.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

template <typename T>
void require_parameter(const MaterialParameterBlock &parameters,
                       std::size_t index, const char *program_name) {
    if (index >= parameters.size() ||
        !std::holds_alternative<T>(parameters[index])) {
        throw std::invalid_argument(std::string(program_name) +
                                    " has an invalid parameter block");
    }
}

void require_size(const MaterialParameterBlock &parameters,
                  std::size_t expected, const char *program_name) {
    if (parameters.size() != expected) {
        throw std::invalid_argument(std::string(program_name) +
                                    " has an invalid parameter count");
    }
}

color evaluate_color(const TextureHandle &texture,
                     const ShaderEvalContext &context) {
    return texture ? texture->evaluate(context).rgb : color(0, 0, 0);
}

color estimate_texture(const TextureHandle &texture) {
    if (!texture) {
        return color(0, 0, 0);
    }
    ShaderEvalContext context;
    context.uv0 = vec2(0.5, 0.5);
    return texture->evaluate(context).rgb;
}

class LambertianProgram final : public MaterialProgram {
  public:
    void evaluate(const ShaderEvalContext &context,
                  const MaterialParameterBlock &parameters, ShaderScratch &,
                  MaterialOutput &output) const override {
        output.reset(context.frame, context.geometry_normal);
        output.bsdf.add_lambertian(
            evaluate_color(parameters.get<TextureHandle>(0), context));
    }

    void validate(const MaterialParameterBlock &parameters) const override {
        require_size(parameters, 1, "LambertianProgram");
        require_parameter<TextureHandle>(parameters, 0,
                                         "LambertianProgram");
    }

    std::size_t max_closures() const override {
        return 1;
    }
};

class MetalProgram final : public MaterialProgram {
  public:
    void evaluate(const ShaderEvalContext &context,
                  const MaterialParameterBlock &parameters, ShaderScratch &,
                  MaterialOutput &output) const override {
        output.reset(context.frame, context.geometry_normal);
        const color &albedo = parameters.get<color>(0);
        const double roughness =
            clamp(parameters.get<double>(1), 0.0, 1.0);
        if (roughness <= 0.001) {
            output.bsdf.add_specular_reflection(albedo);
        } else {
            output.bsdf.add_microfacet_ggx(albedo, roughness, 1.0);
        }
    }

    void validate(const MaterialParameterBlock &parameters) const override {
        require_size(parameters, 2, "MetalProgram");
        require_parameter<color>(parameters, 0, "MetalProgram");
        require_parameter<double>(parameters, 1, "MetalProgram");
    }

    std::size_t max_closures() const override {
        return 1;
    }
};

class DielectricProgram final : public MaterialProgram {
  public:
    void evaluate(const ShaderEvalContext &context,
                  const MaterialParameterBlock &parameters, ShaderScratch &,
                  MaterialOutput &output) const override {
        output.reset(context.frame, context.geometry_normal);
        output.bsdf.add_specular_dielectric(parameters.get<double>(0),
                                            context.front_face);
    }

    void validate(const MaterialParameterBlock &parameters) const override {
        require_size(parameters, 1, "DielectricProgram");
        require_parameter<double>(parameters, 0, "DielectricProgram");
        if (parameters.get<double>(0) <= 1.0) {
            throw std::invalid_argument(
                "DielectricProgram requires ior greater than one");
        }
    }

    std::size_t max_closures() const override {
        return 1;
    }
};

class DiffuseLightProgram final : public MaterialProgram {
  public:
    void evaluate(const ShaderEvalContext &context,
                  const MaterialParameterBlock &parameters, ShaderScratch &,
                  MaterialOutput &output) const override {
        output.reset(context.frame, context.geometry_normal);
        if (context.front_face || parameters.get<bool>(1)) {
            output.set_emission(evaluate_color(
                parameters.get<TextureHandle>(0), context));
        }
    }

    void validate(const MaterialParameterBlock &parameters) const override {
        require_size(parameters, 2, "DiffuseLightProgram");
        require_parameter<TextureHandle>(parameters, 0,
                                         "DiffuseLightProgram");
        require_parameter<bool>(parameters, 1, "DiffuseLightProgram");
    }

    std::size_t max_closures() const override {
        return 0;
    }
};

class PrincipledProgram final : public MaterialProgram {
  public:
    void evaluate(const ShaderEvalContext &context,
                  const MaterialParameterBlock &parameters, ShaderScratch &,
                  MaterialOutput &output) const override {
        NormalMapSettings settings;
        settings.convention =
            parameters.get<std::uint32_t>(8) == 0
                ? NormalMapConvention::OpenGL
                : NormalMapConvention::DirectX;
        settings.strength = parameters.get<double>(9);
        const ShadingFrame frame = apply_normal_map(
            context, parameters.get<TextureHandle>(3), settings);
        output.reset(frame, context.geometry_normal);

        const color base =
            evaluate_color(parameters.get<TextureHandle>(0), context);
        const double roughness =
            clamp(texture_scalar(parameters.get<TextureHandle>(1), context),
                  0.01, 1.0);
        const double metallic =
            clamp(texture_scalar(parameters.get<TextureHandle>(2), context),
                  0.0, 1.0);
        const double diffuse_weight = 1.0 - metallic;
        if (diffuse_weight > 0.0) {
            output.bsdf.add_lambertian(diffuse_weight * base,
                                       diffuse_weight);
        }
        output.bsdf.add_microfacet_ggx(base, roughness, metallic,
                                       0.5 + 0.5 * metallic);

        const TextureHandle &clearcoat =
            parameters.get<TextureHandle>(6);
        if (clearcoat) {
            const double strength =
                clamp(texture_scalar(clearcoat, context), 0.0, 1.0);
            if (strength > 0.0) {
                const TextureHandle &coat_roughness =
                    parameters.get<TextureHandle>(7);
                const double roughness_value =
                    coat_roughness
                        ? clamp(texture_scalar(coat_roughness, context),
                                0.01, 1.0)
                        : 0.1;
                output.bsdf.add_clearcoat_ggx(roughness_value, strength);
            }
        }

        const TextureHandle &emission =
            parameters.get<TextureHandle>(4);
        if (emission &&
            (context.front_face || parameters.get<bool>(10))) {
            output.set_emission(
                parameters.get<double>(5) *
                evaluate_color(emission, context));
        }
    }

    void validate(const MaterialParameterBlock &parameters) const override {
        require_size(parameters, 11, "PrincipledProgram");
        require_parameter<TextureHandle>(parameters, 0,
                                         "PrincipledProgram");
        require_parameter<TextureHandle>(parameters, 1,
                                         "PrincipledProgram");
        require_parameter<TextureHandle>(parameters, 2,
                                         "PrincipledProgram");
        require_parameter<TextureHandle>(parameters, 3,
                                         "PrincipledProgram");
        require_parameter<TextureHandle>(parameters, 4,
                                         "PrincipledProgram");
        require_parameter<double>(parameters, 5, "PrincipledProgram");
        require_parameter<TextureHandle>(parameters, 6,
                                         "PrincipledProgram");
        require_parameter<TextureHandle>(parameters, 7,
                                         "PrincipledProgram");
        require_parameter<std::uint32_t>(parameters, 8,
                                         "PrincipledProgram");
        require_parameter<double>(parameters, 9, "PrincipledProgram");
        require_parameter<bool>(parameters, 10, "PrincipledProgram");
    }

    std::size_t max_closures() const override {
        return 3;
    }
};

class IsotropicProgram final : public MaterialProgram {
  public:
    void evaluate(const ShaderEvalContext &context,
                  const MaterialParameterBlock &parameters, ShaderScratch &,
                  MaterialOutput &output) const override {
        output.reset(context.frame, context.geometry_normal);
        output.bsdf.add_isotropic_phase(
            evaluate_color(parameters.get<TextureHandle>(0), context));
    }

    void validate(const MaterialParameterBlock &parameters) const override {
        require_size(parameters, 1, "IsotropicProgram");
        require_parameter<TextureHandle>(parameters, 0,
                                         "IsotropicProgram");
    }

    std::size_t max_closures() const override {
        return 1;
    }
};

template <typename Program>
std::shared_ptr<const MaterialProgram> program_instance() {
    static const std::shared_ptr<const MaterialProgram> instance =
        std::make_shared<Program>();
    return instance;
}

MaterialHandle make_instance(std::shared_ptr<const MaterialProgram> program,
                             MaterialParameterBlock parameters,
                             MaterialMetadata metadata,
                             MaterialDescription description) {
    return std::make_shared<const MaterialInstance>(
        std::move(program), std::move(parameters), metadata,
        std::move(description));
}

} // namespace

MaterialHandle make_lambertian_material(TextureHandle albedo) {
    LambertianMaterialDescription description{albedo};
    MaterialParameterBlock parameters;
    parameters.add(std::move(albedo));
    return make_instance(program_instance<LambertianProgram>(),
                         std::move(parameters), {}, std::move(description));
}

MaterialHandle make_lambertian_material(const color &albedo) {
    return make_lambertian_material(
        std::make_shared<SolidColorTexture>(albedo));
}

MaterialHandle make_metal_material(const color &albedo, double roughness) {
    const double clamped_roughness = clamp(roughness, 0.0, 1.0);
    MaterialParameterBlock parameters;
    parameters.add(albedo);
    parameters.add(clamped_roughness);
    return make_instance(program_instance<MetalProgram>(),
                         std::move(parameters), {},
                         MetalMaterialDescription{albedo,
                                                  clamped_roughness});
}

MaterialHandle make_dielectric_material(double ior) {
    MaterialParameterBlock parameters;
    parameters.add(ior);
    return make_instance(program_instance<DielectricProgram>(),
                         std::move(parameters), {},
                         DielectricMaterialDescription{ior});
}

MaterialHandle make_diffuse_light_material(TextureHandle emission,
                                           bool double_sided) {
    MaterialMetadata metadata;
    metadata.emissive = true;
    metadata.double_sided = double_sided;
    metadata.emission_estimate = estimate_texture(emission);
    MaterialParameterBlock parameters;
    DiffuseLightMaterialDescription description{emission, double_sided};
    parameters.add(std::move(emission));
    parameters.add(double_sided);
    return make_instance(program_instance<DiffuseLightProgram>(),
                         std::move(parameters), metadata,
                         std::move(description));
}

MaterialHandle make_diffuse_light_material(const color &emission,
                                           bool double_sided) {
    return make_diffuse_light_material(
        std::make_shared<SolidColorTexture>(emission), double_sided);
}

MaterialHandle make_principled_material(
    TextureHandle base_color, TextureHandle roughness, TextureHandle metallic,
    TextureHandle normal_map, TextureHandle emission,
    double emission_strength, TextureHandle clearcoat,
    TextureHandle clearcoat_roughness, NormalMapSettings normal_settings,
    bool double_sided) {
    MaterialMetadata metadata;
    metadata.emissive = emission && emission_strength > 0.0;
    metadata.double_sided = double_sided;
    metadata.emission_estimate =
        emission_strength * estimate_texture(emission);

    PrincipledMaterialDescription description;
    description.base_color = base_color;
    description.roughness = roughness;
    description.metallic = metallic;
    description.normal_map = normal_map;
    description.emission = emission;
    description.clearcoat = clearcoat;
    description.clearcoat_roughness = clearcoat_roughness;
    description.emission_strength = emission_strength;
    description.normal_strength = normal_settings.strength;
    description.normal_convention = static_cast<std::uint32_t>(
        normal_settings.convention == NormalMapConvention::OpenGL ? 0 : 1);
    description.double_sided = double_sided;

    MaterialParameterBlock parameters;
    parameters.add(std::move(base_color));
    parameters.add(std::move(roughness));
    parameters.add(std::move(metallic));
    parameters.add(std::move(normal_map));
    parameters.add(std::move(emission));
    parameters.add(emission_strength);
    parameters.add(std::move(clearcoat));
    parameters.add(std::move(clearcoat_roughness));
    parameters.add(static_cast<std::uint32_t>(
        normal_settings.convention == NormalMapConvention::OpenGL ? 0 : 1));
    parameters.add(normal_settings.strength);
    parameters.add(double_sided);
    return make_instance(program_instance<PrincipledProgram>(),
                         std::move(parameters), metadata,
                         std::move(description));
}

MaterialHandle make_isotropic_material(TextureHandle albedo) {
    IsotropicMaterialDescription description{albedo};
    MaterialParameterBlock parameters;
    parameters.add(std::move(albedo));
    return make_instance(program_instance<IsotropicProgram>(),
                         std::move(parameters), {}, std::move(description));
}

MaterialHandle make_isotropic_material(const color &albedo) {
    return make_isotropic_material(
        std::make_shared<SolidColorTexture>(albedo));
}

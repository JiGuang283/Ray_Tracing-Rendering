#include "material.h"

#include "shading/shader_context.h"

#include <algorithm>

namespace {

ShaderEvalContext make_context(const SurfaceInteraction &surface) {
    return ShaderEvalContext::from_surface(surface, surface.frame.normal);
}

color evaluate_color(const TextureHandle &texture,
                     const ShaderEvalContext &context) {
    return texture ? texture->evaluate(context).rgb : color(0, 0, 0);
}

} // namespace

lambertian::lambertian(const color &a)
    : albedo(make_shared<SolidColorTexture>(a)) {
}

lambertian::lambertian(TextureHandle a) : albedo(std::move(a)) {
}

void lambertian::shade(const SurfaceInteraction &surface,
                       ShadingResult &result) const {
    const ShaderEvalContext context = make_context(surface);
    result.reset(surface.frame);
    result.bsdf.add_lambertian(evaluate_color(albedo, context));
}

metal::metal(const color &a, double f) : albedo(a), fuzz(clamp(f, 0.0, 1.0)) {
}

void metal::shade(const SurfaceInteraction &surface,
                  ShadingResult &result) const {
    result.reset(surface.frame);
    if (fuzz <= 0.001) {
        result.bsdf.add_specular_reflection(albedo);
        return;
    }
    result.bsdf.add_microfacet_ggx(albedo, fuzz, 1.0);
}

dielectric::dielectric(double index_of_refraction)
    : ir(index_of_refraction) {
}

void dielectric::shade(const SurfaceInteraction &surface,
                       ShadingResult &result) const {
    result.reset(surface.frame);
    result.bsdf.add_specular_dielectric(ir, surface.front_face);
}

diffuse_light::diffuse_light(TextureHandle a) : emit(std::move(a)) {
}

diffuse_light::diffuse_light(color c)
    : emit(make_shared<SolidColorTexture>(c)) {
}

void diffuse_light::shade(const SurfaceInteraction &surface,
                          ShadingResult &result) const {
    const ShaderEvalContext context = make_context(surface);
    result.reset(surface.frame);
    if (surface.front_face) {
        result.set_emission(evaluate_color(emit, context));
    }
}

bool diffuse_light::is_emissive() const {
    return true;
}

color diffuse_light::emission_estimate() const {
    return evaluate_color(emit, ShaderEvalContext{});
}

PrincipledMaterial::PrincipledMaterial(
    TextureHandle base_color, TextureHandle roughness, TextureHandle metallic,
    TextureHandle normal_map, TextureHandle emission, double emission_strength,
    TextureHandle clearcoat, TextureHandle clearcoat_roughness,
    NormalMapSettings normal_settings)
    : base_color(std::move(base_color)), roughness(std::move(roughness)),
      metallic(std::move(metallic)), normal_map(std::move(normal_map)),
      emission(std::move(emission)), emission_strength(emission_strength),
      clearcoat(std::move(clearcoat)),
      clearcoat_roughness(std::move(clearcoat_roughness)),
      normal_settings(normal_settings) {
}

void PrincipledMaterial::shade(const SurfaceInteraction &surface,
                               ShadingResult &result) const {
    const ShaderEvalContext context = make_context(surface);
    const ShadingFrame frame =
        apply_normal_map(context, normal_map, normal_settings);
    result.reset(frame);

    const color base = evaluate_color(base_color, context);
    const double rough =
        clamp(texture_scalar(roughness, context), 0.01, 1.0);
    const double metalness =
        clamp(texture_scalar(metallic, context), 0.0, 1.0);

    const double diffuse_weight = 1.0 - metalness;
    if (diffuse_weight > 0.0) {
        result.bsdf.add_lambertian(diffuse_weight * base,
                                   diffuse_weight);
    }
    result.bsdf.add_microfacet_ggx(base, rough, metalness,
                                   0.5 + 0.5 * metalness);

    if (clearcoat) {
        const double coat =
            clamp(texture_scalar(clearcoat, context), 0.0, 1.0);
        if (coat > 0.0) {
            const double coat_roughness =
                clearcoat_roughness
                    ? clamp(texture_scalar(clearcoat_roughness, context),
                            0.01, 1.0)
                    : 0.1;
            result.bsdf.add_clearcoat_ggx(coat_roughness, coat);
        }
    }

    if (emission && surface.front_face) {
        result.set_emission(emission_strength *
                            evaluate_color(emission, context));
    }
}

bool PrincipledMaterial::is_emissive() const {
    return emission != nullptr && emission_strength > 0.0;
}

color PrincipledMaterial::emission_estimate() const {
    if (!emission) {
        return color(0, 0, 0);
    }
    return emission_strength *
           evaluate_color(emission, ShaderEvalContext{});
}

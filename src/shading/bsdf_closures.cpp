#include "shading/bsdf_closures.h"

#include <algorithm>
#include <cmath>

namespace bsdf_closures {
namespace {

double saturate(double value) {
    return clamp(value, 0.0, 1.0);
}

color lerp(const color &a, const color &b, double t) {
    return (1.0 - t) * a + t * b;
}

color fresnel_schlick(double cos_theta, const color &f0) {
    double f = pow(1.0 - saturate(cos_theta), 5.0);
    return f0 + (color(1.0, 1.0, 1.0) - f0) * f;
}

double distribution_ggx(const vec3 &n, const vec3 &h, double roughness) {
    double alpha = roughness * roughness;
    double alpha2 = alpha * alpha;
    double n_dot_h = std::max(dot(n, h), 0.0);
    double n_dot_h2 = n_dot_h * n_dot_h;
    double denom = n_dot_h2 * (alpha2 - 1.0) + 1.0;
    return alpha2 / std::max(pi * denom * denom, 1e-8);
}

double geometry_schlick_ggx(double n_dot_v, double roughness) {
    double r = roughness + 1.0;
    double k = (r * r) / 8.0;
    return n_dot_v / std::max(n_dot_v * (1.0 - k) + k, 1e-8);
}

double geometry_smith(const vec3 &n, const vec3 &wo, const vec3 &wi,
                      double roughness) {
    double n_dot_o = std::max(dot(n, wo), 0.0);
    double n_dot_i = std::max(dot(n, wi), 0.0);
    return geometry_schlick_ggx(n_dot_o, roughness) *
           geometry_schlick_ggx(n_dot_i, roughness);
}

double dielectric_reflectance(double cosine, double ref_idx) {
    double r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow((1.0 - cosine), 5.0);
}

color microfacet_f0(const BSDFClosure &closure) {
    if (closure.type == BSDFClosureType::ClearcoatGGX) {
        return color(0.04, 0.04, 0.04) * closure.tint;
    }
    color dielectric_f0(0.04, 0.04, 0.04);
    return lerp(dielectric_f0, closure.base_color, closure.metallic);
}

bool sample_lambertian(const BSDFClosure &closure,
                       const ShadingFrame &frame, const vec3 &wo,
                       BSDFSample &sampled, RNG &rng) {
    const vec3 &n = frame.normal;
    if (dot(n, wo) <= 0.0) {
        return false;
    }
    sampled.wi = unit_vector(frame.to_world(random_cosine_direction(rng)));
    sampled.f = closure.base_color / pi;
    sampled.pdf = std::max(dot(n, sampled.wi), 0.0) / pi;
    sampled.flags = BSDF_DIFFUSE | BSDF_REFLECTION;
    return sampled.pdf > 0.0;
}

bool sample_specular_reflection(const BSDFClosure &closure,
                                const ShadingFrame &frame, const vec3 &wo,
                                BSDFSample &sampled) {
    const vec3 &n = frame.normal;
    if (dot(n, wo) <= 0.0) {
        return false;
    }
    sampled.wi = unit_vector(reflect(-wo, n));
    if (dot(n, sampled.wi) <= 0.0) {
        return false;
    }
    sampled.f = closure.base_color;
    sampled.pdf = 1.0;
    sampled.flags = BSDF_DELTA | BSDF_REFLECTION;
    return true;
}

bool sample_specular_dielectric(const BSDFClosure &closure,
                                const ShadingFrame &frame, const vec3 &wo,
                                BSDFSample &sampled, RNG &rng) {
    vec3 normal = frame.normal;
    double eta = closure.front_face ? (1.0 / closure.ior) : closure.ior;
    vec3 incident = -wo;
    double cos_theta = fmin(dot(-incident, normal), 1.0);
    double sin_theta = sqrt(std::max(0.0, 1.0 - cos_theta * cos_theta));
    bool cannot_refract = eta * sin_theta > 1.0;

    if (cannot_refract ||
        dielectric_reflectance(cos_theta, eta) > rng.next()) {
        sampled.wi = unit_vector(reflect(incident, normal));
        sampled.flags = BSDF_DELTA | BSDF_REFLECTION;
    } else {
        sampled.wi = unit_vector(refract(incident, normal, eta));
        sampled.flags = BSDF_DELTA | BSDF_TRANSMISSION;
    }
    sampled.f = color(1.0, 1.0, 1.0);
    sampled.pdf = 1.0;
    return true;
}

bool sample_microfacet(const BSDFClosure &closure, const ShadingFrame &frame,
                       const vec3 &wo, BSDFSample &sampled, RNG &rng) {
    const vec3 &n = frame.normal;
    if (dot(n, wo) <= 0.0) {
        return false;
    }

    double r1 = rng.next();
    double r2 = rng.next();
    double alpha = closure.roughness * closure.roughness;
    double phi = 2.0 * pi * r1;
    double cos_theta =
        sqrt((1.0 - r2) / (1.0 + (alpha * alpha - 1.0) * r2));
    double sin_theta = sqrt(std::max(0.0, 1.0 - cos_theta * cos_theta));

    vec3 h_local(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
    vec3 h = unit_vector(frame.to_world(h_local));
    if (dot(h, wo) < 0.0) {
        h = -h;
    }

    sampled.wi = unit_vector(reflect(-wo, h));
    if (dot(n, sampled.wi) <= 0.0) {
        return false;
    }
    sampled.flags = BSDF_GLOSSY | BSDF_REFLECTION;
    sampled.f = eval(closure, frame, wo, sampled.wi);
    sampled.pdf = pdf(closure, frame, wo, sampled.wi);
    return sampled.pdf > 0.0;
}

bool sample_isotropic_phase(const BSDFClosure &closure, BSDFSample &sampled,
                            RNG &rng) {
    sampled.wi = random_unit_vector(rng);
    sampled.f = closure.base_color / (4.0 * pi);
    sampled.pdf = 1.0 / (4.0 * pi);
    sampled.flags = BSDF_PHASE;
    return true;
}

color eval_lambertian(const BSDFClosure &closure, const ShadingFrame &frame,
                      const vec3 &wo, const vec3 &wi) {
    const vec3 &n = frame.normal;
    if (dot(n, wo) <= 0.0 || dot(n, wi) <= 0.0) {
        return color(0, 0, 0);
    }
    return closure.base_color / pi;
}

color eval_microfacet(const BSDFClosure &closure, const ShadingFrame &frame,
                      const vec3 &wo, const vec3 &wi) {
    const vec3 &n = frame.normal;
    double n_dot_o = dot(n, wo);
    double n_dot_i = dot(n, wi);
    if (n_dot_o <= 0.0 || n_dot_i <= 0.0) {
        return color(0, 0, 0);
    }

    vec3 h = unit_vector(wo + wi);
    color f = fresnel_schlick(std::max(dot(h, wo), 0.0),
                              microfacet_f0(closure));
    double d = distribution_ggx(n, h, closure.roughness);
    double g = geometry_smith(n, wo, wi, closure.roughness);
    double denom = std::max(4.0 * n_dot_o * n_dot_i, 1e-8);
    return (d * g / denom) * f;
}

double pdf_lambertian(const ShadingFrame &frame, const vec3 &wo,
                      const vec3 &wi) {
    const vec3 &n = frame.normal;
    if (dot(n, wo) <= 0.0 || dot(n, wi) <= 0.0) {
        return 0.0;
    }
    return std::max(dot(n, wi), 0.0) / pi;
}

double pdf_microfacet(const BSDFClosure &closure, const ShadingFrame &frame,
                      const vec3 &wo, const vec3 &wi) {
    const vec3 &n = frame.normal;
    if (dot(n, wo) <= 0.0 || dot(n, wi) <= 0.0) {
        return 0.0;
    }
    vec3 h = unit_vector(wo + wi);
    double h_dot_o = std::max(dot(h, wo), 0.0);
    if (h_dot_o <= 0.0) {
        return 0.0;
    }
    double d = distribution_ggx(n, h, closure.roughness);
    double n_dot_h = std::max(dot(n, h), 0.0);
    return d * n_dot_h / std::max(4.0 * h_dot_o, 1e-8);
}

} // namespace

double luminance(const color &c) {
    return 0.2126 * c.x() + 0.7152 * c.y() + 0.0722 * c.z();
}

bool sample(const BSDFClosure &closure, const ShadingFrame &frame,
            const vec3 &wo, BSDFSample &sampled, RNG &rng) {
    switch (closure.type) {
    case BSDFClosureType::Lambertian:
        return sample_lambertian(closure, frame, wo, sampled, rng);
    case BSDFClosureType::SpecularReflection:
        return sample_specular_reflection(closure, frame, wo, sampled);
    case BSDFClosureType::SpecularDielectric:
        return sample_specular_dielectric(closure, frame, wo, sampled, rng);
    case BSDFClosureType::MicrofacetGGX:
    case BSDFClosureType::ClearcoatGGX:
        return sample_microfacet(closure, frame, wo, sampled, rng);
    case BSDFClosureType::IsotropicPhase:
        return sample_isotropic_phase(closure, sampled, rng);
    }
    return false;
}

color eval(const BSDFClosure &closure, const ShadingFrame &frame,
           const vec3 &wo, const vec3 &wi) {
    switch (closure.type) {
    case BSDFClosureType::Lambertian:
        return eval_lambertian(closure, frame, wo, wi);
    case BSDFClosureType::MicrofacetGGX:
    case BSDFClosureType::ClearcoatGGX:
        return eval_microfacet(closure, frame, wo, wi);
    case BSDFClosureType::IsotropicPhase:
        return closure.base_color / (4.0 * pi);
    case BSDFClosureType::SpecularReflection:
    case BSDFClosureType::SpecularDielectric:
        return color(0, 0, 0);
    }
    return color(0, 0, 0);
}

double pdf(const BSDFClosure &closure, const ShadingFrame &frame,
           const vec3 &wo, const vec3 &wi) {
    switch (closure.type) {
    case BSDFClosureType::Lambertian:
        return pdf_lambertian(frame, wo, wi);
    case BSDFClosureType::MicrofacetGGX:
    case BSDFClosureType::ClearcoatGGX:
        return pdf_microfacet(closure, frame, wo, wi);
    case BSDFClosureType::IsotropicPhase:
        return 1.0 / (4.0 * pi);
    case BSDFClosureType::SpecularReflection:
    case BSDFClosureType::SpecularDielectric:
        return 0.0;
    }
    return 0.0;
}

} // namespace bsdf_closures

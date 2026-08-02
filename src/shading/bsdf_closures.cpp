#include "bsdf_closures.h"

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace bsdf_closures {
namespace {

double saturate(double value) {
    return clamp(value, 0.0, 1.0);
}

color fresnel_schlick(double cosine, const color &f0) {
    const double factor = std::pow(1.0 - saturate(cosine), 5.0);
    return f0 + (color(1, 1, 1) - f0) * factor;
}

double fresnel_dielectric(double cosine_i, double eta_i, double eta_t) {
    cosine_i = clamp(cosine_i, 0.0, 1.0);
    const double sin_i =
        std::sqrt(std::max(0.0, 1.0 - cosine_i * cosine_i));
    const double sin_t = eta_i / eta_t * sin_i;
    if (sin_t >= 1.0) {
        return 1.0;
    }

    const double cosine_t =
        std::sqrt(std::max(0.0, 1.0 - sin_t * sin_t));
    const double parallel =
        ((eta_t * cosine_i) - (eta_i * cosine_t)) /
        ((eta_t * cosine_i) + (eta_i * cosine_t));
    const double perpendicular =
        ((eta_i * cosine_i) - (eta_t * cosine_t)) /
        ((eta_i * cosine_i) + (eta_t * cosine_t));
    return 0.5 *
           (parallel * parallel + perpendicular * perpendicular);
}

double ggx_alpha(double roughness) {
    return std::max(roughness * roughness, 1e-4);
}

double distribution_ggx(double n_dot_h, double alpha) {
    const double alpha2 = alpha * alpha;
    const double cosine2 = n_dot_h * n_dot_h;
    const double denominator = cosine2 * (alpha2 - 1.0) + 1.0;
    return alpha2 /
           std::max(pi * denominator * denominator, 1e-12);
}

double smith_g1(double n_dot_v, double alpha) {
    if (n_dot_v <= 0.0) {
        return 0.0;
    }
    const double cosine2 = n_dot_v * n_dot_v;
    const double tangent2 =
        std::max(0.0, 1.0 - cosine2) / std::max(cosine2, 1e-12);
    return 2.0 /
           (1.0 + std::sqrt(1.0 + alpha * alpha * tangent2));
}

bool finite_vector(const vec3 &value) {
    return std::isfinite(value.x()) && std::isfinite(value.y()) &&
           std::isfinite(value.z());
}

vec3 sample_visible_ggx_normal(const vec3 &local_wo, double alpha,
                               const vec2 &u) {
    const vec3 stretched =
        unit_vector(vec3(alpha * local_wo.x(), alpha * local_wo.y(),
                         local_wo.z()));
    const double lensq = stretched.x() * stretched.x() +
                         stretched.y() * stretched.y();
    const vec3 tangent1 =
        lensq > 1e-20
            ? vec3(-stretched.y(), stretched.x(), 0.0) / std::sqrt(lensq)
            : vec3(1, 0, 0);
    const vec3 tangent2 = cross(stretched, tangent1);

    const double radius = std::sqrt(clamp(u.x(), 0.0, 1.0));
    const double phi = 2.0 * pi * clamp(u.y(), 0.0, 1.0);
    const double disk_x = radius * std::cos(phi);
    double disk_y = radius * std::sin(phi);
    const double blend = 0.5 * (1.0 + stretched.z());
    disk_y = (1.0 - blend) *
                 std::sqrt(std::max(0.0, 1.0 - disk_x * disk_x)) +
             blend * disk_y;

    const double disk_z = std::sqrt(
        std::max(0.0, 1.0 - disk_x * disk_x - disk_y * disk_y));
    const vec3 visible_normal =
        disk_x * tangent1 + disk_y * tangent2 + disk_z * stretched;
    return unit_vector(vec3(alpha * visible_normal.x(),
                            alpha * visible_normal.y(),
                            std::max(visible_normal.z(), 1e-8)));
}

template <typename Closure>
std::optional<BSDFSample> sample_impl(const Closure &, const ShadingFrame &,
                                      const vec3 &, RNG &) {
    return std::nullopt;
}

template <>
std::optional<BSDFSample>
sample_impl(const LambertianClosure &closure, const ShadingFrame &frame,
            const vec3 &wo, RNG &rng) {
    if (dot(frame.normal, wo) <= 0.0) {
        return std::nullopt;
    }

    BSDFSample sample;
    sample.wi = unit_vector(frame.to_world(random_cosine_direction(rng)));
    sample.f = closure.albedo / pi;
    sample.pdf = std::max(dot(frame.normal, sample.wi), 0.0) / pi;
    sample.flags = BSDF_DIFFUSE | BSDF_REFLECTION;
    return sample.pdf > 0.0 ? std::optional<BSDFSample>(sample)
                            : std::nullopt;
}

template <>
std::optional<BSDFSample>
sample_impl(const SpecularReflectionClosure &closure,
            const ShadingFrame &frame, const vec3 &wo, RNG &) {
    const double cosine_o = dot(frame.normal, wo);
    if (cosine_o <= 0.0) {
        return std::nullopt;
    }

    BSDFSample sample;
    sample.wi = unit_vector(reflect(-wo, frame.normal));
    const double cosine_i = std::abs(dot(frame.normal, sample.wi));
    if (cosine_i <= 0.0) {
        return std::nullopt;
    }
    sample.f = closure.reflectance / cosine_i;
    sample.pdf = 1.0;
    sample.flags = BSDF_DELTA | BSDF_REFLECTION;
    return sample;
}

template <>
std::optional<BSDFSample>
sample_impl(const SpecularDielectricClosure &closure,
            const ShadingFrame &frame, const vec3 &wo, RNG &rng) {
    const double cosine_o = dot(frame.normal, wo);
    if (cosine_o <= 0.0) {
        return std::nullopt;
    }

    const double eta_i = closure.front_face ? 1.0 : closure.ior;
    const double eta_t = closure.front_face ? closure.ior : 1.0;
    const double eta_ratio = eta_i / eta_t;
    const double reflectance =
        fresnel_dielectric(cosine_o, eta_i, eta_t);
    const bool reflect_event = rng.next() < reflectance;

    BSDFSample sample;
    if (reflect_event) {
        sample.wi = unit_vector(reflect(-wo, frame.normal));
        const double cosine_i = std::abs(dot(frame.normal, sample.wi));
        sample.f = color(reflectance, reflectance, reflectance) /
                   std::max(cosine_i, 1e-12);
        sample.pdf = reflectance;
        sample.flags = BSDF_DELTA | BSDF_REFLECTION;
    } else {
        sample.wi =
            unit_vector(refract(-wo, frame.normal, eta_ratio));
        const double cosine_i = std::abs(dot(frame.normal, sample.wi));
        const double transmittance = 1.0 - reflectance;
        const double radiance_scale = eta_ratio * eta_ratio;
        sample.f =
            color(transmittance, transmittance, transmittance) *
            radiance_scale / std::max(cosine_i, 1e-12);
        sample.pdf = transmittance;
        sample.flags = BSDF_DELTA | BSDF_TRANSMISSION;
        sample.eta = eta_t / eta_i;
    }
    return sample.pdf > 0.0 ? std::optional<BSDFSample>(sample)
                            : std::nullopt;
}

template <typename Closure>
std::optional<BSDFSample>
sample_ggx(const Closure &closure, const color &f0,
           const ShadingFrame &frame, const vec3 &wo, RNG &rng) {
    const vec3 local_wo = frame.to_local(wo);
    if (local_wo.z() <= 0.0 || !finite_vector(local_wo)) {
        return std::nullopt;
    }

    const double alpha = ggx_alpha(closure.roughness);
    const vec3 local_half = sample_visible_ggx_normal(
        unit_vector(local_wo), alpha, vec2(rng.next(), rng.next()));
    const vec3 half_vector = unit_vector(frame.to_world(local_half));

    BSDFSample sample;
    sample.wi = unit_vector(reflect(-wo, half_vector));
    if (!finite_vector(sample.wi) || dot(frame.normal, sample.wi) <= 0.0) {
        return std::nullopt;
    }

    const double n_dot_o = dot(frame.normal, wo);
    const double n_dot_i = dot(frame.normal, sample.wi);
    const double n_dot_h = std::max(dot(frame.normal, half_vector), 0.0);
    const double o_dot_h = std::max(dot(wo, half_vector), 0.0);
    const double d = distribution_ggx(n_dot_h, alpha);
    const double g =
        smith_g1(n_dot_o, alpha) * smith_g1(n_dot_i, alpha);
    sample.f = fresnel_schlick(o_dot_h, f0) * d * g /
               std::max(4.0 * n_dot_o * n_dot_i, 1e-12);
    sample.pdf = d * smith_g1(n_dot_o, alpha) /
                 std::max(4.0 * n_dot_o, 1e-12);
    sample.flags = BSDF_GLOSSY | BSDF_REFLECTION;
    if (!std::isfinite(sample.pdf) || sample.pdf <= 0.0 ||
        !finite_vector(sample.f)) {
        return std::nullopt;
    }
    return sample;
}

template <>
std::optional<BSDFSample>
sample_impl(const GGXReflectionClosure &closure,
            const ShadingFrame &frame, const vec3 &wo, RNG &rng) {
    return sample_ggx(closure, closure.f0, frame, wo, rng);
}

template <>
std::optional<BSDFSample>
sample_impl(const ClearcoatGGXClosure &closure,
            const ShadingFrame &frame, const vec3 &wo, RNG &rng) {
    return sample_ggx(closure, color(0.04, 0.04, 0.04), frame, wo, rng);
}

template <>
std::optional<BSDFSample>
sample_impl(const IsotropicPhaseClosure &closure, const ShadingFrame &,
            const vec3 &, RNG &rng) {
    BSDFSample sample;
    sample.wi = random_unit_vector(rng);
    sample.f = closure.albedo / (4.0 * pi);
    sample.pdf = 1.0 / (4.0 * pi);
    sample.flags = BSDF_PHASE;
    return sample;
}

template <typename Closure>
color eval_impl(const Closure &, const ShadingFrame &, const vec3 &,
                const vec3 &) {
    return color(0, 0, 0);
}

template <>
color eval_impl(const LambertianClosure &closure,
                const ShadingFrame &frame, const vec3 &wo,
                const vec3 &wi) {
    if (dot(frame.normal, wo) <= 0.0 || dot(frame.normal, wi) <= 0.0) {
        return color(0, 0, 0);
    }
    return closure.albedo / pi;
}

template <typename Closure>
color eval_ggx(const Closure &closure, const color &f0,
               const ShadingFrame &frame, const vec3 &wo,
               const vec3 &wi) {
    const double n_dot_o = dot(frame.normal, wo);
    const double n_dot_i = dot(frame.normal, wi);
    if (n_dot_o <= 0.0 || n_dot_i <= 0.0 || (wo + wi).near_zero()) {
        return color(0, 0, 0);
    }

    const vec3 half_vector = unit_vector(wo + wi);
    const double n_dot_h = std::max(dot(frame.normal, half_vector), 0.0);
    const double o_dot_h = std::max(dot(wo, half_vector), 0.0);
    const double alpha = ggx_alpha(closure.roughness);
    const double d = distribution_ggx(n_dot_h, alpha);
    const double g =
        smith_g1(n_dot_o, alpha) * smith_g1(n_dot_i, alpha);
    return fresnel_schlick(o_dot_h, f0) * d * g /
           std::max(4.0 * n_dot_o * n_dot_i, 1e-12);
}

template <>
color eval_impl(const GGXReflectionClosure &closure,
                const ShadingFrame &frame, const vec3 &wo,
                const vec3 &wi) {
    return eval_ggx(closure, closure.f0, frame, wo, wi);
}

template <>
color eval_impl(const ClearcoatGGXClosure &closure,
                const ShadingFrame &frame, const vec3 &wo,
                const vec3 &wi) {
    return eval_ggx(closure, color(0.04, 0.04, 0.04), frame, wo, wi);
}

template <>
color eval_impl(const IsotropicPhaseClosure &closure,
                const ShadingFrame &, const vec3 &, const vec3 &) {
    return closure.albedo / (4.0 * pi);
}

template <typename Closure>
double pdf_impl(const Closure &, const ShadingFrame &, const vec3 &,
                const vec3 &) {
    return 0.0;
}

template <>
double pdf_impl(const LambertianClosure &, const ShadingFrame &frame,
                const vec3 &wo, const vec3 &wi) {
    if (dot(frame.normal, wo) <= 0.0 || dot(frame.normal, wi) <= 0.0) {
        return 0.0;
    }
    return dot(frame.normal, wi) / pi;
}

template <typename Closure>
double pdf_ggx(const Closure &closure, const ShadingFrame &frame,
               const vec3 &wo, const vec3 &wi) {
    if (dot(frame.normal, wo) <= 0.0 || dot(frame.normal, wi) <= 0.0 ||
        (wo + wi).near_zero()) {
        return 0.0;
    }
    const vec3 half_vector = unit_vector(wo + wi);
    const double o_dot_h = std::max(dot(wo, half_vector), 0.0);
    if (o_dot_h <= 0.0) {
        return 0.0;
    }
    const double n_dot_h = std::max(dot(frame.normal, half_vector), 0.0);
    const double d =
        distribution_ggx(n_dot_h, ggx_alpha(closure.roughness));
    const double result =
        d * smith_g1(dot(frame.normal, wo),
                     ggx_alpha(closure.roughness)) /
        std::max(4.0 * dot(frame.normal, wo), 1e-12);
    return std::isfinite(result) ? result : 0.0;
}

template <>
double pdf_impl(const GGXReflectionClosure &closure,
                const ShadingFrame &frame, const vec3 &wo,
                const vec3 &wi) {
    return pdf_ggx(closure, frame, wo, wi);
}

template <>
double pdf_impl(const ClearcoatGGXClosure &closure,
                const ShadingFrame &frame, const vec3 &wo,
                const vec3 &wi) {
    return pdf_ggx(closure, frame, wo, wi);
}

template <>
double pdf_impl(const IsotropicPhaseClosure &, const ShadingFrame &,
                const vec3 &, const vec3 &) {
    return 1.0 / (4.0 * pi);
}

} // namespace

double luminance(const color &value) {
    return 0.2126 * value.x() + 0.7152 * value.y() +
           0.0722 * value.z();
}

bool is_delta(const ClosureVariant &closure) {
    return std::holds_alternative<SpecularReflectionClosure>(closure) ||
           std::holds_alternative<SpecularDielectricClosure>(closure);
}

bool is_phase(const ClosureVariant &closure) {
    return std::holds_alternative<IsotropicPhaseClosure>(closure);
}

std::optional<BSDFSample> sample(const ClosureVariant &closure,
                                 const ShadingFrame &frame, const vec3 &wo,
                                 RNG &rng) {
    return std::visit(
        [&](const auto &typed_closure) {
            return sample_impl(typed_closure, frame, wo, rng);
        },
        closure);
}

color eval(const ClosureVariant &closure, const ShadingFrame &frame,
           const vec3 &wo, const vec3 &wi) {
    return std::visit(
        [&](const auto &typed_closure) {
            return eval_impl(typed_closure, frame, wo, wi);
        },
        closure);
}

double pdf(const ClosureVariant &closure, const ShadingFrame &frame,
           const vec3 &wo, const vec3 &wi) {
    return std::visit(
        [&](const auto &typed_closure) {
            return pdf_impl(typed_closure, frame, wo, wi);
        },
        closure);
}

} // namespace bsdf_closures

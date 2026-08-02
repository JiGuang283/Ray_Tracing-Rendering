#include "bsdf.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace {

color lerp(const color &a, const color &b, double t) {
    return (1.0 - t) * a + t * b;
}

bool is_black(const color &value) {
    return value.x() <= 0.0 && value.y() <= 0.0 && value.z() <= 0.0;
}

} // namespace

BSDF::BSDF() : m_frame(vec3(0, 0, 1)), m_geometry_normal(0, 0, 1) {
}

BSDF::BSDF(const ShadingFrame &frame)
    : m_frame(frame), m_geometry_normal(frame.normal) {
}

BSDF::BSDF(const ShadingFrame &frame, const vec3 &geometry_normal)
    : m_frame(frame), m_geometry_normal(unit_vector(geometry_normal)) {
}

void BSDF::reset(const ShadingFrame &frame) {
    reset(frame, frame.normal);
}

void BSDF::reset(const ShadingFrame &frame, const vec3 &geometry_normal) {
    m_frame = frame;
    m_geometry_normal = unit_vector(geometry_normal);
    m_count = 0;
}

bool BSDF::empty() const {
    return m_count == 0;
}

std::size_t BSDF::size() const {
    return m_count;
}

void BSDF::add_lambertian(const color &albedo, double sample_weight) {
    if (is_black(albedo)) {
        return;
    }
    add_closure(LambertianClosure{albedo}, color(1, 1, 1), sample_weight);
}

void BSDF::add_specular_reflection(const color &reflectance,
                                   double sample_weight) {
    if (is_black(reflectance)) {
        return;
    }
    add_closure(SpecularReflectionClosure{reflectance}, color(1, 1, 1),
                sample_weight);
}

void BSDF::add_specular_dielectric(double ior, bool front_face,
                                   double sample_weight) {
    add_closure(SpecularDielectricClosure{std::max(ior, 1.0001), front_face},
                color(1, 1, 1), sample_weight);
}

void BSDF::add_microfacet_ggx(const color &base_color, double roughness,
                              double metallic, double sample_weight) {
    const double metalness = clamp(metallic, 0.0, 1.0);
    const color f0 =
        lerp(color(0.04, 0.04, 0.04), base_color, metalness);
    add_closure(GGXReflectionClosure{f0, clamp(roughness, 0.01, 1.0)},
                color(1, 1, 1), sample_weight);
}

void BSDF::add_clearcoat_ggx(double roughness, double strength) {
    const double coat = clamp(strength, 0.0, 1.0);
    if (coat <= 0.0) {
        return;
    }
    add_closure(ClearcoatGGXClosure{clamp(roughness, 0.01, 1.0)},
                color(coat, coat, coat), coat);
}

void BSDF::add_isotropic_phase(const color &albedo, double sample_weight) {
    if (is_black(albedo)) {
        return;
    }
    add_closure(IsotropicPhaseClosure{albedo}, color(1, 1, 1),
                sample_weight);
}

std::optional<BSDFSample> BSDF::sample(const vec3 &wo, RNG &rng) const {
    if (empty()) {
        return std::nullopt;
    }

    const double total_weight = total_sample_weight();
    if (total_weight <= 0.0) {
        return std::nullopt;
    }

    const std::size_t index = choose_closure(rng.next());
    const ClosureEntry &entry = m_closures[index];
    const double selection_pdf = entry.sample_weight / total_weight;
    auto candidate =
        bsdf_closures::sample(entry.closure, m_frame, unit_vector(wo), rng);
    if (!candidate || candidate->pdf <= 0.0) {
        return std::nullopt;
    }
    if (!valid_event(unit_vector(wo), candidate->wi, candidate->flags)) {
        return std::nullopt;
    }

    if (candidate->is_delta()) {
        candidate->f *= entry.weight;
        candidate->pdf *= selection_pdf;
        return candidate;
    }

    candidate->f = eval(wo, candidate->wi);
    candidate->pdf = pdf(wo, candidate->wi);
    if (candidate->pdf <= 0.0) {
        return std::nullopt;
    }
    return candidate;
}

color BSDF::eval(const vec3 &wo, const vec3 &wi) const {
    color result(0, 0, 0);
    const vec3 unit_wo = unit_vector(wo);
    const vec3 unit_wi = unit_vector(wi);
    if (!is_phase() &&
        !valid_event(unit_wo, unit_wi,
                     BSDF_REFLECTION | BSDF_GLOSSY)) {
        return result;
    }
    for (std::size_t i = 0; i < m_count; ++i) {
        const ClosureEntry &entry = m_closures[i];
        result += entry.weight *
                  bsdf_closures::eval(entry.closure, m_frame, unit_wo,
                                      unit_wi);
    }
    return result;
}

double BSDF::pdf(const vec3 &wo, const vec3 &wi) const {
    const double total_weight = total_sample_weight();
    if (total_weight <= 0.0) {
        return 0.0;
    }

    double result = 0.0;
    const vec3 unit_wo = unit_vector(wo);
    const vec3 unit_wi = unit_vector(wi);
    if (!is_phase() &&
        !valid_event(unit_wo, unit_wi,
                     BSDF_REFLECTION | BSDF_GLOSSY)) {
        return 0.0;
    }
    for (std::size_t i = 0; i < m_count; ++i) {
        const ClosureEntry &entry = m_closures[i];
        const double selection_pdf = entry.sample_weight / total_weight;
        result += selection_pdf *
                  bsdf_closures::pdf(entry.closure, m_frame, unit_wo,
                                     unit_wi);
    }
    return result;
}

bool BSDF::is_phase() const {
    return m_count == 1 && bsdf_closures::is_phase(m_closures[0].closure);
}

double BSDF::abs_cos_theta(const vec3 &w) const {
    return std::abs(dot(unit_vector(w), m_frame.normal));
}

const ShadingFrame &BSDF::frame() const {
    return m_frame;
}

void BSDF::add_closure(ClosureVariant closure, const color &weight,
                       double sample_weight) {
    if (sample_weight <= 0.0 || is_black(weight)) {
        return;
    }
    if (m_count >= kMaxClosures) {
        throw std::overflow_error("BSDF closure capacity exceeded");
    }
    m_closures[m_count++] =
        ClosureEntry{std::move(closure), weight, sample_weight};
}

double BSDF::total_sample_weight() const {
    double total = 0.0;
    for (std::size_t i = 0; i < m_count; ++i) {
        total += std::max(m_closures[i].sample_weight, 0.0);
    }
    return total;
}

std::size_t BSDF::choose_closure(double u) const {
    const double target = u * total_sample_weight();
    double accumulated = 0.0;
    for (std::size_t i = 0; i < m_count; ++i) {
        accumulated += std::max(m_closures[i].sample_weight, 0.0);
        if (target < accumulated) {
            return i;
        }
    }
    return m_count - 1;
}

bool BSDF::valid_event(const vec3 &wo, const vec3 &wi, int flags) const {
    if (has_flag(flags, BSDF_PHASE)) {
        return true;
    }
    const double outgoing_side = dot(wo, m_geometry_normal);
    const double incoming_side = dot(wi, m_geometry_normal);
    if (std::abs(outgoing_side) <= 1e-12 ||
        std::abs(incoming_side) <= 1e-12) {
        return false;
    }
    if (has_flag(flags, BSDF_TRANSMISSION)) {
        return outgoing_side * incoming_side < 0.0;
    }
    return outgoing_side * incoming_side > 0.0;
}

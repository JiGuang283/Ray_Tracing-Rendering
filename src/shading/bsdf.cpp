#include "shading/bsdf.h"

#include "shading/bsdf_closures.h"

#include <algorithm>

BSDF::BSDF() : m_frame(vec3(0, 0, 1)) {
}

BSDF::BSDF(const ShadingFrame &frame) : m_frame(frame) {
}

void BSDF::reset(const ShadingFrame &frame) {
    m_frame = frame;
    m_count = 0;
}

bool BSDF::empty() const {
    return m_count == 0;
}

void BSDF::add_lambertian(const color &albedo, double sample_weight) {
    BSDFClosure closure;
    closure.type = BSDFClosureType::Lambertian;
    closure.base_color = albedo;
    closure.sample_weight =
        std::max(sample_weight,
                 std::max(0.0, bsdf_closures::luminance(albedo)));
    add_closure(closure);
}

void BSDF::add_specular_reflection(const color &albedo, double sample_weight) {
    BSDFClosure closure;
    closure.type = BSDFClosureType::SpecularReflection;
    closure.base_color = albedo;
    closure.sample_weight =
        std::max(sample_weight,
                 std::max(0.0, bsdf_closures::luminance(albedo)));
    add_closure(closure);
}

void BSDF::add_specular_dielectric(double ior, bool front_face,
                                   double sample_weight) {
    BSDFClosure closure;
    closure.type = BSDFClosureType::SpecularDielectric;
    closure.ior = std::max(ior, 1e-4);
    closure.front_face = front_face;
    closure.sample_weight = std::max(sample_weight, 1e-4);
    add_closure(closure);
}

void BSDF::add_microfacet_ggx(const color &base_color, double roughness,
                              double metallic, double sample_weight) {
    BSDFClosure closure;
    closure.type = BSDFClosureType::MicrofacetGGX;
    closure.base_color = base_color;
    closure.roughness = clamp(roughness, 0.01, 1.0);
    closure.metallic = clamp(metallic, 0.0, 1.0);
    closure.sample_weight =
        std::max(sample_weight,
                 std::max(0.0, bsdf_closures::luminance(base_color)));
    add_closure(closure);
}

void BSDF::add_clearcoat_ggx(double roughness, double sample_weight) {
    BSDFClosure closure;
    closure.type = BSDFClosureType::ClearcoatGGX;
    closure.base_color = color(1.0, 1.0, 1.0);
    closure.tint = color(1.0, 1.0, 1.0);
    closure.roughness = clamp(roughness, 0.01, 1.0);
    closure.metallic = 0.0;
    closure.sample_weight = std::max(sample_weight, 1e-4);
    add_closure(closure);
}

void BSDF::add_isotropic_phase(const color &albedo, double sample_weight) {
    BSDFClosure closure;
    closure.type = BSDFClosureType::IsotropicPhase;
    closure.base_color = albedo;
    closure.sample_weight =
        std::max(sample_weight,
                 std::max(0.0, bsdf_closures::luminance(albedo)));
    add_closure(closure);
}

bool BSDF::sample(const vec3 &wo, BSDFSample &sampled, RNG &rng) const {
    if (empty()) {
        return false;
    }

    int start = choose_closure(rng.next());
    for (int attempt = 0; attempt < m_count; ++attempt) {
        int index = (start + attempt) % m_count;
        BSDFSample candidate;
        if (!sample_closure(m_closures[index], unit_vector(wo), candidate,
                            rng)) {
            continue;
        }

        if (candidate.is_delta()) {
            sampled = candidate;
            sampled.pdf = 1.0;
            return true;
        }

        sampled = candidate;
        sampled.f = eval(wo, sampled.wi);
        sampled.pdf = pdf(wo, sampled.wi);
        return sampled.pdf > 1e-8 || sampled.is_phase();
    }

    return false;
}

color BSDF::eval(const vec3 &wo, const vec3 &wi) const {
    color result(0, 0, 0);
    vec3 unit_wo = unit_vector(wo);
    vec3 unit_wi = unit_vector(wi);
    for (int i = 0; i < m_count; ++i) {
        result += eval_closure(m_closures[i], unit_wo, unit_wi);
    }
    return result;
}

double BSDF::pdf(const vec3 &wo, const vec3 &wi) const {
    double total_weight = total_sample_weight();
    if (total_weight <= 0.0) {
        return 0.0;
    }

    double result = 0.0;
    vec3 unit_wo = unit_vector(wo);
    vec3 unit_wi = unit_vector(wi);
    for (int i = 0; i < m_count; ++i) {
        const auto &closure = m_closures[i];
        double selection_pdf = closure.sample_weight / total_weight;
        result += selection_pdf * pdf_closure(closure, unit_wo, unit_wi);
    }
    return result;
}

bool BSDF::is_phase() const {
    return m_count == 1 &&
           m_closures[0].type == BSDFClosureType::IsotropicPhase;
}

double BSDF::abs_cos_theta(const vec3 &w) const {
    return std::abs(dot(unit_vector(w), m_frame.normal));
}

const ShadingFrame &BSDF::frame() const {
    return m_frame;
}

bool BSDF::add_closure(const BSDFClosure &closure) {
    if (m_count >= kMaxClosures) {
        return false;
    }
    if (closure.sample_weight <= 0.0) {
        return false;
    }
    if (closure.base_color.length_squared() <= 0.0 &&
        (closure.type == BSDFClosureType::Lambertian ||
         closure.type == BSDFClosureType::SpecularReflection ||
         closure.type == BSDFClosureType::IsotropicPhase)) {
        return false;
    }
    m_closures[m_count++] = closure;
    return true;
}

double BSDF::total_sample_weight() const {
    double total = 0.0;
    for (int i = 0; i < m_count; ++i) {
        total += std::max(m_closures[i].sample_weight, 0.0);
    }
    return total;
}

int BSDF::choose_closure(double u) const {
    double total = total_sample_weight();
    if (total <= 0.0) {
        return 0;
    }

    double target = u * total;
    double accum = 0.0;
    for (int i = 0; i < m_count; ++i) {
        accum += std::max(m_closures[i].sample_weight, 0.0);
        if (target <= accum) {
            return i;
        }
    }
    return m_count - 1;
}

bool BSDF::sample_closure(const BSDFClosure &closure, const vec3 &wo,
                          BSDFSample &sampled, RNG &rng) const {
    return bsdf_closures::sample(closure, m_frame, wo, sampled, rng);
}

color BSDF::eval_closure(const BSDFClosure &closure, const vec3 &wo,
                         const vec3 &wi) const {
    return bsdf_closures::eval(closure, m_frame, wo, wi);
}

double BSDF::pdf_closure(const BSDFClosure &closure, const vec3 &wo,
                         const vec3 &wi) const {
    return bsdf_closures::pdf(closure, m_frame, wo, wi);
}

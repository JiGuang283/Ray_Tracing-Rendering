#ifndef BSDF_H
#define BSDF_H

#include "interaction.h"
#include "rtweekend.h"

#include <array>

enum class BSDFClosureType {
    Lambertian,
    SpecularReflection,
    SpecularDielectric,
    MicrofacetGGX,
    ClearcoatGGX,
    IsotropicPhase
};

struct BSDFClosure {
    BSDFClosureType type = BSDFClosureType::Lambertian;
    color base_color = color(0, 0, 0);
    color tint = color(1, 1, 1);
    double roughness = 0.5;
    double metallic = 0.0;
    double ior = 1.5;
    bool front_face = true;
    double sample_weight = 1.0;
};

class BSDF {
  public:
    BSDF();
    explicit BSDF(const ShadingFrame &frame);

    void reset(const ShadingFrame &frame);
    bool empty() const;

    void add_lambertian(const color &albedo, double sample_weight = 1.0);
    void add_specular_reflection(const color &albedo,
                                 double sample_weight = 1.0);
    void add_specular_dielectric(double ior, bool front_face = true,
                                 double sample_weight = 1.0);
    void add_microfacet_ggx(const color &base_color, double roughness,
                            double metallic, double sample_weight = 1.0);
    void add_clearcoat_ggx(double roughness, double sample_weight = 1.0);
    void add_isotropic_phase(const color &albedo, double sample_weight = 1.0);

    bool sample(const vec3 &wo, BSDFSample &sampled, RNG &rng) const;
    color eval(const vec3 &wo, const vec3 &wi) const;
    double pdf(const vec3 &wo, const vec3 &wi) const;

    bool is_phase() const;
    double abs_cos_theta(const vec3 &w) const;
    const ShadingFrame &frame() const;

  private:
    static constexpr int kMaxClosures = 8;

    bool add_closure(const BSDFClosure &closure);
    double total_sample_weight() const;
    int choose_closure(double u) const;

    bool sample_closure(const BSDFClosure &closure, const vec3 &wo,
                        BSDFSample &sampled, RNG &rng) const;
    color eval_closure(const BSDFClosure &closure, const vec3 &wo,
                       const vec3 &wi) const;
    double pdf_closure(const BSDFClosure &closure, const vec3 &wo,
                       const vec3 &wi) const;

    ShadingFrame m_frame;
    std::array<BSDFClosure, kMaxClosures> m_closures;
    int m_count = 0;
};

#endif

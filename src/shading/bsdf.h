#ifndef BSDF_H
#define BSDF_H

#include "shading/bsdf_closures.h"

#include <array>
#include <cstddef>
#include <optional>

struct ClosureEntry {
    ClosureVariant closure;
    color weight{1, 1, 1};
    double sample_weight = 1.0;
};

class BSDF {
  public:
    static constexpr std::size_t kMaxClosures = 8;

    BSDF();
    explicit BSDF(const ShadingFrame &frame);

    void reset(const ShadingFrame &frame);
    bool empty() const;
    std::size_t size() const;

    void add_lambertian(const color &albedo, double sample_weight = 1.0);
    void add_specular_reflection(const color &reflectance,
                                 double sample_weight = 1.0);
    void add_specular_dielectric(double ior, bool front_face = true,
                                 double sample_weight = 1.0);
    void add_microfacet_ggx(const color &base_color, double roughness,
                            double metallic, double sample_weight = 1.0);
    void add_clearcoat_ggx(double roughness, double strength = 1.0);
    void add_isotropic_phase(const color &albedo,
                             double sample_weight = 1.0);

    std::optional<BSDFSample> sample(const vec3 &wo, RNG &rng) const;
    color eval(const vec3 &wo, const vec3 &wi) const;
    double pdf(const vec3 &wo, const vec3 &wi) const;

    bool is_phase() const;
    double abs_cos_theta(const vec3 &w) const;
    const ShadingFrame &frame() const;

  private:
    void add_closure(ClosureVariant closure, const color &weight,
                     double sample_weight);
    double total_sample_weight() const;
    std::size_t choose_closure(double u) const;

    ShadingFrame m_frame;
    std::array<ClosureEntry, kMaxClosures> m_closures;
    std::size_t m_count = 0;
};

#endif

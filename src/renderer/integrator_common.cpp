#include "integrator_common.h"

#include <algorithm>

namespace integrator_common {

ShadedSurface shade_surface(const hit_record &rec, const ray &r) {
    ShadedSurface shaded;
    shaded.surface = SurfaceInteraction(rec);
    shaded.wo = -unit_vector(r.direction());
    shaded.time = r.time();
    if (rec.mat_ptr) {
        rec.mat_ptr->shade(shaded.surface, shaded.shading);
    }
    return shaded;
}

double power_heuristic(double pdf_a, double pdf_b) {
    double a2 = pdf_a * pdf_a;
    double b2 = pdf_b * pdf_b;
    double denom = a2 + b2;
    return denom > 0 ? a2 / denom : 0.0;
}

color clamp_radiance(const color &L, double max_value) {
    if (L.x() > max_value || L.y() > max_value || L.z() > max_value) {
        double max_c = std::max({L.x(), L.y(), L.z()});
        if (max_c > max_value) {
            return L * (max_value / max_c);
        }
    }
    return L;
}

double scattering_cos_factor(const ShadingResult &shading,
                             const BSDFSample &sample) {
    return sample.is_phase() ? 1.0 : shading.bsdf.abs_cos_theta(sample.wi);
}

color scattering_weight(const ShadingResult &shading,
                        const BSDFSample &sample) {
    if (sample.pdf <= 0.0) {
        return color(0, 0, 0);
    }
    return sample.f * scattering_cos_factor(shading, sample) / sample.pdf;
}

bool visible(const hittable &scene, const ray &shadow_ray,
             double max_distance, RNG &rng) {
    hit_record shadow_rec;
    return !scene.hit(shadow_ray, 0.001, max_distance - 0.001, shadow_rec,
                      rng);
}

color sample_direct_lighting(const ShadedSurface &shaded,
                             const hittable &scene,
                             const LightSampler &light_sampler, RNG &rng,
                             bool use_mis) {
    if (light_sampler.empty() || shaded.shading.bsdf.empty()) {
        return color(0, 0, 0);
    }

    double light_select_pdf = 0.0;
    LightSample ls =
        light_sampler.sample(shaded.surface.p, rng, light_select_pdf);
    if (ls.pdf <= 0.0 || light_select_pdf <= 0.0 ||
        ls.Li.length_squared() <= 0.0) {
        return color(0, 0, 0);
    }

    if (!visible(scene, shaded.surface.spawn_ray(ls.wi, shaded.time), ls.dist,
                 rng)) {
        return color(0, 0, 0);
    }

    color f = shaded.shading.bsdf.eval(shaded.wo, ls.wi);
    double cos_theta = shaded.shading.bsdf.is_phase()
                           ? 1.0
                           : shaded.shading.bsdf.abs_cos_theta(ls.wi);

    if (ls.is_delta || !use_mis) {
        return f * ls.Li * cos_theta / (ls.pdf * light_select_pdf);
    }

    double bsdf_pdf = shaded.shading.bsdf.pdf(shaded.wo, ls.wi);
    double light_pdf = ls.pdf * light_select_pdf;
    double mis_weight = power_heuristic(light_pdf, bsdf_pdf);
    return f * ls.Li * cos_theta * mis_weight / light_pdf;
}

} // namespace integrator_common

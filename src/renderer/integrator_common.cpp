#include "integrator_common.h"

#include <algorithm>
#include <cmath>

namespace integrator_common {

ShadedSurface shade_surface(const hit_record &rec, const ray &r,
                            ShaderScratch &scratch) {
    ShadedSurface shaded;
    shaded.surface = SurfaceInteraction(rec);
    shaded.wo = -unit_vector(r.direction());
    shaded.time = r.time();
    if (rec.mat_ptr) {
        const ShaderEvalContext context = ShaderEvalContext::from_surface(
            shaded.surface, shaded.wo, shaded.time);
        rec.mat_ptr->evaluate(context, scratch, shaded.shading);
    }
    return shaded;
}

double power_heuristic(double pdf_a, double pdf_b) {
    double a2 = pdf_a * pdf_a;
    double b2 = pdf_b * pdf_b;
    double denom = a2 + b2;
    return denom > 0 ? a2 / denom : 0.0;
}

double scattering_cos_factor(const MaterialOutput &shading,
                             const BSDFSample &sample) {
    return sample.is_phase() ? 1.0 : shading.bsdf.abs_cos_theta(sample.wi);
}

color scattering_weight(const MaterialOutput &shading,
                        const BSDFSample &sample) {
    if (sample.pdf <= 0.0) {
        return color(0, 0, 0);
    }
    return sample.f * scattering_cos_factor(shading, sample) / sample.pdf;
}

bool visible(const hittable &scene, const ray &shadow_ray,
             double max_distance, RNG &rng,
             std::uint64_t *shadow_rays) {
    if (shadow_rays != nullptr) {
        ++*shadow_rays;
    }
    return !scene.occluded(shadow_ray, 0.001, max_distance - 0.001, rng);
}

namespace {

bool finite_color(const color &value) {
    return std::isfinite(value.x()) && std::isfinite(value.y()) &&
           std::isfinite(value.z());
}

bool finite_direction(const vec3 &value) {
    return std::isfinite(value.x()) && std::isfinite(value.y()) &&
           std::isfinite(value.z()) && value.length_squared() > 0.0;
}

color evaluate_direct_sample(const ShadedSurface &shaded,
                             const hittable &scene, const LightSample &sample,
                             double selection_pdf,
                             bool has_bsdf_competitor, RNG &rng,
                             bool use_mis,
                             std::uint64_t *shadow_rays) {
    if (sample.pdf <= 0.0 || selection_pdf <= 0.0 ||
        !std::isfinite(sample.pdf) || !std::isfinite(selection_pdf) ||
        !finite_color(sample.Li) || !finite_direction(sample.wi) ||
        sample.Li.length_squared() <= 0.0) {
        return color(0, 0, 0);
    }

    const color f = shaded.shading.bsdf.eval(shaded.wo, sample.wi);
    const double cos_theta = shaded.shading.bsdf.is_phase()
                                 ? 1.0
                                 : shaded.shading.bsdf.abs_cos_theta(sample.wi);
    if (!finite_color(f) || f.length_squared() <= 0.0 || cos_theta <= 0.0 ||
        !std::isfinite(cos_theta)) {
        return color(0, 0, 0);
    }

    if (!visible(scene, shaded.surface.spawn_ray(sample.wi, shaded.time),
                 sample.dist, rng, shadow_rays)) {
        return color(0, 0, 0);
    }

    const double light_pdf = sample.pdf * selection_pdf;
    if (sample.is_delta || !use_mis || !has_bsdf_competitor) {
        return f * sample.Li * cos_theta / light_pdf;
    }

    const double bsdf_pdf =
        shaded.shading.bsdf.pdf(shaded.wo, sample.wi);
    const double mis_weight = power_heuristic(light_pdf, bsdf_pdf);
    return f * sample.Li * cos_theta * mis_weight / light_pdf;
}

} // namespace

color sample_direct_lighting(const ShadedSurface &shaded,
                             const hittable &scene,
                             const LightSampler &light_sampler, RNG &rng,
                             bool use_mis,
                             std::uint64_t *shadow_rays) {
    if (light_sampler.empty() || shaded.shading.bsdf.empty()) {
        return color(0, 0, 0);
    }

    color result(0, 0, 0);
    for (const auto &light : light_sampler.delta_lights()) {
        const LightSample sample =
            light->sample(shaded.surface.p, vec2(rng.next(), rng.next()));
        result += evaluate_direct_sample(shaded, scene, sample, 1.0, false,
                                         rng, use_mis, shadow_rays);
    }

    if (light_sampler.has_non_delta_lights()) {
        const SelectedLightSample selected =
            light_sampler.sample_non_delta(shaded.surface.p, rng);
        result += evaluate_direct_sample(
            shaded, scene, selected.sample, selected.selection_pdf,
            selected.has_bsdf_competitor, rng, use_mis, shadow_rays);
    }
    return result;
}

} // namespace integrator_common

#include "cpu_path_integrator.h"

#include "integrator_common.h"
#include "light_sampler.h"
#include "rtweekend.h"

#include <algorithm>
#include <stdexcept>

namespace {

color miss_radiance(const ray &current_ray, const color &background,
                    const LightSampler &light_sampler,
                    const IntegratorPolicy &policy, int depth,
                    bool delta_bounce, double previous_bsdf_pdf) {
    if (!policy.uses_direct_lighting()) {
        return background;
    }

    bool found_environment = false;
    color radiance = light_sampler.environment_radiance(
        current_ray, found_environment);
    if (!found_environment) {
        return background;
    }
    if (policy.uses_mis() && depth != 0 && !delta_bounce) {
        const double light_pdf = light_sampler.pdf(
            current_ray.origin(), current_ray.direction());
        radiance *= integrator_common::power_heuristic(
            previous_bsdf_pdf, light_pdf);
    }
    return radiance;
}

color emitted_radiance(const integrator_common::ShadedSurface &shaded,
                       const ray &current_ray,
                       const LightSampler &light_sampler,
                       const IntegratorPolicy &policy, int depth,
                       bool delta_bounce, double previous_bsdf_pdf) {
    if (!shaded.shading.has_emission) {
        return color(0, 0, 0);
    }
    if (policy.uses_direct_lighting() && !policy.uses_mis() && depth != 0 &&
        !delta_bounce) {
        return color(0, 0, 0);
    }
    if (!policy.uses_mis() || depth == 0 || delta_bounce ||
        light_sampler.empty()) {
        return shaded.shading.emission;
    }

    const double light_pdf = light_sampler.pdf(
        current_ray.origin(), current_ray.direction());
    return shaded.shading.emission *
           integrator_common::power_heuristic(previous_bsdf_pdf, light_pdf);
}

} // namespace

CpuPathIntegrator::CpuPathIntegrator(IntegratorPolicy policy)
    : m_policy(policy) {
    if (!valid_integrator_policy(policy)) {
        throw std::invalid_argument("invalid CPU integrator policy");
    }
}

CpuPathIntegrator::CpuPathIntegrator(IntegratorKind kind)
    : CpuPathIntegrator(integrator_policy(kind)) {
}

void CpuPathIntegrator::set_max_depth(int depth) {
    if (depth <= 0) {
        throw std::invalid_argument("maximum path depth must be positive");
    }
    m_max_depth = depth;
}

color CpuPathIntegrator::Li(const ray &r, const hittable &scene,
                            const color &background,
                            IntegratorContext &context) const {
    static const LightSampler empty_sampler({});
    const LightSampler &light_sampler = context.light_sampler
                                            ? *context.light_sampler
                                            : empty_sampler;
    RNG &rng = context.rng;
    color throughput(1, 1, 1);
    color radiance(0, 0, 0);
    ray current_ray = r;
    bool delta_bounce = false;
    double previous_bsdf_pdf = 0.0;
    double eta_scale = 1.0;

    for (int depth = 0; depth < m_max_depth; ++depth) {
        hit_record record;
        if (!scene.hit(current_ray, 0.001, infinity, record, rng)) {
            radiance += throughput *
                        miss_radiance(current_ray, background, light_sampler,
                                      m_policy, depth, delta_bounce,
                                      previous_bsdf_pdf);
            break;
        }

        const auto shaded = integrator_common::shade_surface(
            record, current_ray, context.shader_scratch);
        radiance += throughput *
                    emitted_radiance(shaded, current_ray, light_sampler,
                                     m_policy, depth, delta_bounce,
                                     previous_bsdf_pdf);

        if (m_policy.uses_direct_lighting() &&
            !shaded.shading.bsdf.empty() && !light_sampler.empty()) {
            radiance +=
                throughput * integrator_common::sample_direct_lighting(
                                 shaded, scene, light_sampler, rng,
                                 m_policy.uses_mis());
        }

        if (depth + 1 >= m_max_depth) {
            break;
        }
        const auto sample = shaded.shading.bsdf.sample(shaded.wo, rng);
        if (!sample || sample->pdf < 1e-8) {
            break;
        }

        delta_bounce = sample->is_delta();
        previous_bsdf_pdf = delta_bounce ? 0.0 : sample->pdf;
        throughput *=
            integrator_common::scattering_weight(shaded.shading, *sample);
        if (sample->is_transmission()) {
            eta_scale *= sample->eta * sample->eta;
        }
        current_ray =
            shaded.surface.spawn_ray(sample->wi, current_ray.time());

        if (m_policy.uses_russian_roulette() &&
            depth >= static_cast<int>(m_policy.rr_start_depth)) {
            const color compensated = eta_scale * throughput;
            double survival = std::max(
                {compensated.x(), compensated.y(), compensated.z()});
            survival = clamp(survival, m_policy.rr_min_survival, 0.95);
            if (rng.next() > survival) {
                break;
            }
            throughput /= survival;
        }
    }
    return radiance;
}

std::shared_ptr<Integrator> make_cpu_integrator(IntegratorKind kind) {
    return std::make_shared<CpuPathIntegrator>(kind);
}

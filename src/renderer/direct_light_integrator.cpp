#include "direct_light_integrator.h"

#include <algorithm>
#include <cmath>

#include "integrator_common.h"
#include "light_sampler.h"
#include "material.h"
#include "rtweekend.h"

DirectLightIntegrator::DirectLightIntegrator() = default;

void DirectLightIntegrator::set_max_depth(int depth) {
    m_max_depth = depth;
}

void DirectLightIntegrator::set_rr_start_depth(int depth) {
    m_rr_start_depth = depth;
}

color DirectLightIntegrator::Li(const ray &r, const hittable &scene,
                                const color &background,
                                IntegratorContext &context) const {
    return Li(r, scene, background, {}, context);
}

color DirectLightIntegrator::Li(
    const ray &r, const hittable &scene, const color &background,
    const std::vector<shared_ptr<Light>> &lights,
    IntegratorContext &context) const {
    RNG &rng = context.rng;
    color throughput(1.0, 1.0, 1.0);
    color L(0.0, 0.0, 0.0);
    ray current_ray = r;
    bool delta_bounce = false;
    double eta_scale = 1.0;
    LightSampler light_sampler(lights);

    for (int depth = 0; depth < m_max_depth; ++depth) {
        hit_record rec;
        if (!scene.hit(current_ray, 0.001, infinity, rec, rng)) {
            bool found_env = false;
            color env = light_sampler.environment_radiance(current_ray,
                                                           found_env);
            if (!found_env) {
                L += throughput * background;
            } else {
                L += throughput * env;
            }
            break;
        }

        auto shaded = integrator_common::shade_surface(
            rec, current_ray, context.shader_scratch);

        if (depth == 0 || delta_bounce) {
            L += throughput * shaded.shading.emission;
        }

        if (!light_sampler.empty()) {
            L += throughput * integrator_common::sample_direct_lighting(
                                  shaded, scene, light_sampler, rng, false);
        }

        auto bs = shaded.shading.bsdf.sample(shaded.wo, rng);
        if (!bs) {
            break;
        }

        if (bs->pdf < 1e-8) {
            break;
        }

        delta_bounce = bs->is_delta();

        throughput *=
            integrator_common::scattering_weight(shaded.shading, *bs);
        if (bs->is_transmission()) {
            eta_scale *= bs->eta * bs->eta;
        }

        current_ray =
            shaded.surface.spawn_ray(bs->wi, current_ray.time());

        if (depth >= m_rr_start_depth) {
            const color rr_throughput = eta_scale * throughput;
            double p_survive =
                std::max({rr_throughput.x(), rr_throughput.y(),
                          rr_throughput.z()});
            p_survive = clamp(p_survive, 0.05, 0.95);
            if (rng.next() > p_survive) {
                break;
            }
            throughput /= p_survive;
        }
    }
    return L;
}

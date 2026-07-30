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
                                const color &background) const {
    RNG rng(make_thread_seed());
    return Li(r, scene, background, {}, rng);
}

color DirectLightIntegrator::Li(
    const ray &r, const hittable &scene, const color &background,
    const std::vector<shared_ptr<Light>> &lights) const {
    RNG rng(make_thread_seed());
    return Li(r, scene, background, lights, rng);
}

color DirectLightIntegrator::Li(const ray &r, const hittable &scene,
                                const color &background, RNG &rng) const {
    return Li(r, scene, background, {}, rng);
}

color DirectLightIntegrator::Li(
    const ray &r, const hittable &scene, const color &background,
    const std::vector<shared_ptr<Light>> &lights, RNG &rng) const {
    color throughput(1.0, 1.0, 1.0);
    color L(0.0, 0.0, 0.0);
    ray current_ray = r;
    bool delta_bounce = false;
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

        auto shaded = integrator_common::shade_surface(rec, current_ray);

        if (depth == 0 || delta_bounce) {
            L += throughput * shaded.shading.emission;
        }

        if (!light_sampler.empty()) {
            L += throughput *
                 integrator_common::clamp_radiance(
                     integrator_common::sample_direct_lighting(
                         shaded, scene, light_sampler, rng, false));
        }

        BSDFSample bs;
        if (shaded.shading.bsdf.empty() ||
            !shaded.shading.bsdf.sample(shaded.wo, bs, rng)) {
            break;
        }

        if (bs.pdf < 1e-8 && !bs.is_delta()) {
            break;
        }

        delta_bounce = bs.is_delta();

        throughput *= integrator_common::scattering_weight(shaded.shading, bs);

        current_ray = shaded.surface.spawn_ray(bs.wi, current_ray.time());

        if (depth >= m_rr_start_depth) {
            double p_survive =
                std::max({throughput.x(), throughput.y(), throughput.z()});
            p_survive = clamp(p_survive, 0.05, 0.95);
            if (rng.next() > p_survive) {
                break;
            }
            throughput /= p_survive;
        }
    }
    return L;
}

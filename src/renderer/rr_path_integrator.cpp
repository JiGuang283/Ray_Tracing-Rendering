#include "rr_path_integrator.h"

#include <algorithm>
#include <cmath>

#include "integrator_common.h"
#include "material.h"
#include "rtweekend.h"

RRPathInterator::RRPathInterator() = default;

void RRPathInterator::set_max_depth(int depth) {
    m_max_depth = depth;
}

void RRPathInterator::set_rr_start_depth(int depth) {
    m_rr_start_depth = depth;
}

color RRPathInterator::Li(const ray &r, const hittable &scene,
                          const color &background) const {
    RNG rng(make_thread_seed());
    return Li(r, scene, background, rng);
}

color RRPathInterator::Li(const ray &r, const hittable &scene,
                          const color &background, RNG &rng) const {
    color throughput(1.0, 1.0, 1.0);
    color L(0.0, 0.0, 0.0);
    ray current_ray = r;

    for (int depth = 0; depth < m_max_depth; ++depth) {
        hit_record rec;

        if (!scene.hit(current_ray, 0.001, infinity, rec, rng)) {
            L += throughput * background;
            break;
        }

        auto shaded = integrator_common::shade_surface(rec, current_ray);
        L += throughput * shaded.shading.emission;

        BSDFSample bs;
        if (shaded.shading.bsdf.empty() ||
            !shaded.shading.bsdf.sample(shaded.wo, bs, rng)) {
            break;
        }
        if (bs.pdf < 1e-8 && !bs.is_delta()) {
            break;
        }

        throughput *= integrator_common::scattering_weight(shaded.shading, bs);

        if (depth >= m_rr_start_depth) {
            double p_survive =
                std::max({throughput.x(), throughput.y(), throughput.z()});
            p_survive = clamp(p_survive, 0.005, 0.95);

            if (rng.next() > p_survive) {
                break;
            }
            throughput /= p_survive;
        }
        current_ray = shaded.surface.spawn_ray(bs.wi, current_ray.time());
    }
    return L;
}

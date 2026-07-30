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
                          const color &background,
                          IntegratorContext &context) const {
    RNG &rng = context.rng;
    color throughput(1.0, 1.0, 1.0);
    color L(0.0, 0.0, 0.0);
    ray current_ray = r;
    double eta_scale = 1.0;

    for (int depth = 0; depth < m_max_depth; ++depth) {
        hit_record rec;

        if (!scene.hit(current_ray, 0.001, infinity, rec, rng)) {
            L += throughput * background;
            break;
        }

        auto shaded = integrator_common::shade_surface(
            rec, current_ray, context.shader_scratch);
        L += throughput * shaded.shading.emission;

        auto bs = shaded.shading.bsdf.sample(shaded.wo, rng);
        if (!bs) {
            break;
        }
        if (bs->pdf < 1e-8) {
            break;
        }

        throughput *=
            integrator_common::scattering_weight(shaded.shading, *bs);
        if (bs->is_transmission()) {
            eta_scale *= bs->eta * bs->eta;
        }

        if (depth >= m_rr_start_depth) {
            const color rr_throughput = eta_scale * throughput;
            double p_survive =
                std::max({rr_throughput.x(), rr_throughput.y(),
                          rr_throughput.z()});
            p_survive = clamp(p_survive, 0.005, 0.95);

            if (rng.next() > p_survive) {
                break;
            }
            throughput /= p_survive;
        }
        current_ray =
            shaded.surface.spawn_ray(bs->wi, current_ray.time());
    }
    return L;
}

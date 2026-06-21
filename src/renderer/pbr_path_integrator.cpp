#include "pbr_path_integrator.h"

#include <algorithm>
#include <cmath>

#include "material.h"
#include "rtweekend.h"

PBRPathIntegrator::PBRPathIntegrator() = default;

void PBRPathIntegrator::set_max_depth(int depth) {
    m_max_depth = depth;
}

void PBRPathIntegrator::set_rr_start_depth(int depth) {
    m_rr_start_depth = depth;
}

color PBRPathIntegrator::Li(const ray &r, const hittable &scene,
                            const color &background) const {
    RNG rng(make_thread_seed());
    return Li(r, scene, background, rng);
}

color PBRPathIntegrator::Li(const ray &r, const hittable &scene,
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

        vec3 wo = -unit_vector(current_ray.direction());

        color emitted = rec.mat_ptr->emitted(rec, wo);
        L += throughput * emitted;

        BSDFSample bs;

        if (!rec.mat_ptr->sample(rec, wo, bs, rng)) {
            break;
        }

        if (bs.pdf < 1e-8 && !bs.is_specular) {
            break;
        }

        double cos_theta = std::abs(dot(bs.wi, rec.normal));

        if (bs.is_specular) {
            throughput *= bs.f;
        } else {
            throughput *= bs.f * cos_theta / bs.pdf;
        }

        current_ray = ray(rec.p, bs.wi, current_ray.time());

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

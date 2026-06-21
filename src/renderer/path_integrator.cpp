#include "path_integrator.h"

#include <cmath>

#include "material.h"
#include "rtweekend.h"

PathIntegrator::PathIntegrator() = default;

void PathIntegrator::set_max_depth(int depth) {
    m_max_depth = depth;
}

color PathIntegrator::Li(const ray &r, const hittable &scene,
                         const color &background) const {
    RNG rng(make_thread_seed());
    return Li(r, scene, background, rng);
}

color PathIntegrator::Li(const ray &r, const hittable &scene,
                         const color &background, RNG &rng) const {
    return Li_internal(r, scene, background, m_max_depth, rng);
}

color PathIntegrator::Li_internal(const ray &r, const hittable &scene,
                                  const color &background, int depth,
                                  RNG &rng) const {
    hit_record rec;

    if (depth <= 0) {
        return color(0, 0, 0);
    }

    if (!scene.hit(r, 0.001, infinity, rec, rng)) {
        return background;
    }

    vec3 wo = -unit_vector(r.direction());
    color emitted = rec.mat_ptr->emitted(rec, wo);

    BSDFSample bs;
    if (!rec.mat_ptr->sample(rec, wo, bs, rng)) {
        return emitted;
    }
    if (bs.pdf < 1e-8 && !bs.is_specular) {
        return emitted;
    }

    color throughput = bs.f;
    if (!bs.is_specular) {
        double cos_theta = std::abs(dot(bs.wi, rec.normal));
        throughput *= cos_theta / bs.pdf;
    }

    ray scattered(rec.p, bs.wi, r.time());
    return emitted + throughput * Li_internal(scattered, scene, background,
                                              depth - 1, rng);
}

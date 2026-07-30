#include "path_integrator.h"

#include <cmath>

#include "integrator_common.h"
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

    auto shaded = integrator_common::shade_surface(rec, r);
    color emitted = shaded.shading.emission;

    auto bs = shaded.shading.bsdf.sample(shaded.wo, rng);
    if (!bs) {
        return emitted;
    }
    if (bs->pdf < 1e-8) {
        return emitted;
    }

    color throughput =
        integrator_common::scattering_weight(shaded.shading, *bs);

    ray scattered = shaded.surface.spawn_ray(bs->wi, r.time());
    return emitted + throughput * Li_internal(scattered, scene, background,
                                              depth - 1, rng);
}

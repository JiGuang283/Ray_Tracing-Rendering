#ifndef PATH_INTEGRATOR_H
#define PATH_INTEGRATOR_H

#include "integrator.h"
#include "material.h"
#include "rtweekend.h"

class PathIntegrator : public Integrator {
  public:
    PathIntegrator() = default;

    void set_max_depth(int depth = 50) override {
        m_max_depth = depth;
    }

    virtual color Li(const ray &r, const hittable &scene,
                     const color &background) const override {
        RNG rng(make_thread_seed());
        return Li(r, scene, background, rng);
    }

    virtual color Li(const ray &r, const hittable &scene,
                     const color &background, RNG &rng) const override {
        return Li_internal(r, scene, background, m_max_depth, rng);
    }

  private:
    color Li_internal(const ray &r, const hittable &scene,
                      const color &background, int depth, RNG &rng) const {
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
    int m_max_depth = 50;
};

#endif

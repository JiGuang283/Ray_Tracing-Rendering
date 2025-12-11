#ifndef PBR_PATH_INTEGRATOR_H
#define PBR_PATH_INTEGRATOR_H

#include "integrator.h"
#include "material.h"
#include "rtweekend.h"
#include <algorithm>

class PBRPathIntegrator : public Integrator {
  public:
    PBRPathIntegrator() = default;

    void set_max_depth(int depth = 50) override {
        m_max_depth = depth;
    }

    void set_rr_start_depth(int depth) {
        m_rr_start_depth = depth;
    }

    virtual color Li(const ray &r, const hittable &scene,
                     const color &background) const override {
        color throughput(1.0, 1.0, 1.0);
        color L(0.0, 0.0, 0.0);
        ray current_ray = r;

        for (int depth = 0; depth < m_max_depth; ++depth) {
            hit_record rec;

            if (!scene.hit(current_ray, 0.001, infinity, rec)) {
                L += throughput * background;
                break;
            }

            vec3 wo = -unit_vector(current_ray.direction());

            color emitted = rec.mat_ptr->emitted(rec, wo);
            L += throughput * emitted;

            BSDFSample bs;

            if (!rec.mat_ptr->sample(rec, wo, bs)) {
                break; // 采样失败（例如被吸收），停止追踪
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

                if (random_double() > p_survive) {
                    break;
                }
                throughput /= p_survive;
            }
        }
        return L;
    }

  private:
    int m_max_depth = 50;
    int m_rr_start_depth = 3;
};

#endif
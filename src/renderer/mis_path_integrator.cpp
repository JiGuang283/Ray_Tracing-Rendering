#include "mis_path_integrator.h"

#include <algorithm>
#include <cmath>

#include "material.h"
#include "rtweekend.h"

MISPathIntegrator::MISPathIntegrator() = default;

void MISPathIntegrator::set_max_depth(int depth) {
    m_max_depth = depth;
}

void MISPathIntegrator::set_rr_start_depth(int depth) {
    m_rr_start_depth = depth;
}

color MISPathIntegrator::Li(const ray &r, const hittable &scene,
                            const color &background) const {
    RNG rng(make_thread_seed());
    return Li(r, scene, background, {}, rng);
}

color MISPathIntegrator::Li(
    const ray &r, const hittable &scene, const color &background,
    const std::vector<shared_ptr<Light>> &lights) const {
    RNG rng(make_thread_seed());
    return Li(r, scene, background, lights, rng);
}

color MISPathIntegrator::Li(const ray &r, const hittable &scene,
                            const color &background, RNG &rng) const {
    return Li(r, scene, background, {}, rng);
}

color MISPathIntegrator::Li(
    const ray &r, const hittable &scene, const color &background,
    const std::vector<shared_ptr<Light>> &lights, RNG &rng) const {
    color throughput(1.0, 1.0, 1.0);
    color L(0.0, 0.0, 0.0);
    ray current_ray = r;
    bool specular_bounce = false;
    double prev_bsdf_pdf = 0.0;

    for (int depth = 0; depth < m_max_depth; ++depth) {
        hit_record rec;

        if (!scene.hit(current_ray, 0.001, infinity, rec, rng)) {
            color env_L(0, 0, 0);
            bool found_env = false;

            for (const auto &light : lights) {
                if (light->is_infinite()) {
                    env_L += light->Le(current_ray);
                    found_env = true;
                }
            }

            if (!found_env) {
                L += throughput * background;
            } else {
                if (depth == 0 || specular_bounce) {
                    L += throughput * env_L;
                } else {
                    double light_pdf = 0.0;
                    double light_select_pdf = 1.0 / lights.size();
                    for (const auto &light : lights) {
                        light_pdf += light->pdf(current_ray.origin(),
                                                current_ray.direction()) *
                                     light_select_pdf;
                    }
                    double mis_weight =
                        power_heuristic(prev_bsdf_pdf, light_pdf);
                    L += throughput * env_L * mis_weight;
                }
            }
            break;
        }

        vec3 wo = -unit_vector(current_ray.direction());

        color emitted = rec.mat_ptr->emitted(rec, wo);
        if (emitted.length_squared() > 0) {
            color L_emit(0, 0, 0);
            if (depth == 0 || specular_bounce) {
                L_emit = throughput * emitted;
            } else if (!lights.empty()) {
                double light_pdf =
                    compute_light_pdf(rec, wo, lights, current_ray);
                double mis_weight = power_heuristic(prev_bsdf_pdf, light_pdf);
                L_emit = throughput * emitted * mis_weight;
            } else {
                L_emit = throughput * emitted;
            }

            if (depth == 0) {
                L += L_emit;
            } else {
                L += clamp_radiance(L_emit);
            }
        }

        specular_bounce = rec.mat_ptr->is_specular();

        if (!specular_bounce && !lights.empty()) {
            color L_direct =
                throughput * sample_lights_mis(rec, wo, scene, lights, rng);
            L += clamp_radiance(L_direct);
        }

        BSDFSample bs;
        if (!rec.mat_ptr->sample(rec, wo, bs, rng)) {
            break;
        }

        if (bs.pdf < 1e-8 && !bs.is_specular) {
            break;
        }

        specular_bounce = bs.is_specular;
        prev_bsdf_pdf = bs.is_specular ? 0.0 : bs.pdf;

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

color MISPathIntegrator::clamp_radiance(const color &L, double max_value) {
    if (L.x() > max_value || L.y() > max_value || L.z() > max_value) {
        double max_c = std::max({L.x(), L.y(), L.z()});
        if (max_c > max_value) {
            return L * (max_value / max_c);
        }
    }
    return L;
}

double MISPathIntegrator::power_heuristic(double pdf_a, double pdf_b) {
    double a2 = pdf_a * pdf_a;
    double b2 = pdf_b * pdf_b;
    double denom = a2 + b2;
    return denom > 0 ? a2 / denom : 0.0;
}

double MISPathIntegrator::compute_light_pdf(
    const hit_record &rec, const vec3 &wo,
    const std::vector<shared_ptr<Light>> &lights,
    const ray &current_ray) const {
    double total_pdf = 0.0;
    double light_select_pdf = 1.0 / lights.size();

    for (const auto &light : lights) {
        total_pdf += light->pdf(current_ray.origin(), current_ray.direction()) *
                     light_select_pdf;
    }

    return total_pdf;
}

color MISPathIntegrator::sample_lights_mis(
    const hit_record &rec, const vec3 &wo, const hittable &scene,
    const std::vector<shared_ptr<Light>> &lights, RNG &rng) const {
    if (lights.empty()) {
        return color(0, 0, 0);
    }

    color L_direct(0, 0, 0);

    int light_idx = rng.next_int(0, static_cast<int>(lights.size()) - 1);
    const auto &light = lights[light_idx];
    double light_select_pdf = 1.0 / lights.size();

    vec2 u(rng.next(), rng.next());
    LightSample ls = light->sample(rec.p, u);

    if (ls.pdf > 0 && ls.Li.length_squared() > 0) {
        ray shadow_ray(rec.p, ls.wi, 0);
        hit_record shadow_rec;
        bool in_shadow =
            scene.hit(shadow_ray, 0.001, ls.dist - 0.001, shadow_rec, rng);

        if (!in_shadow) {
            color f = rec.mat_ptr->eval(rec, wo, ls.wi);
            double cos_theta = std::abs(dot(ls.wi, rec.normal));

            if (ls.is_delta) {
                L_direct += f * ls.Li * cos_theta / light_select_pdf;
            } else {
                double bsdf_pdf = rec.mat_ptr->pdf(rec, wo, ls.wi);
                double light_pdf = ls.pdf * light_select_pdf;
                double mis_weight = power_heuristic(light_pdf, bsdf_pdf);

                L_direct += f * ls.Li * cos_theta * mis_weight / light_pdf;
            }
        }
    }

    return L_direct;
}

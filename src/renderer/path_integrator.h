#ifndef PATH_INTEGRATOR_H
#define PATH_INTEGRATOR_H

#include "integrator.h"
#include "material.h"
#include "rtweekend.h"
#include "light.h" // 必须包含 light 基类
#include <vector>

class PathIntegrator : public Integrator {
public:
    PathIntegrator() = default;

    void set_max_depth(int depth = 50) override {
        m_max_depth = depth;
    }

    // ★★★ 新增：设置光源列表的函数 ★★★
    void set_lights(const std::vector<shared_ptr<Light>>& lights) {
        m_lights = lights;
    }

    // 重写基类的 Li 接口
    virtual color Li(const ray& r, const hittable& scene,
        const color& background) const override {
        // ★★★ 这里不再传 empty_lights，而是传成员变量 m_lights ★★★
        return Li_internal(r, scene, m_lights, background, m_max_depth);
    }

private:
    // 成员变量
    int m_max_depth = 50; // 给个默认值
    std::vector<shared_ptr<Light>> m_lights; // ★★★ 存储光源列表 ★★★
    color Li_internal(const ray& r, const hittable& scene,
        const std::vector<shared_ptr<Light>>& lights,
        const color& background, int depth) const {
        hit_record rec;

        if (depth <= 0) {
            return color(0, 0, 0);
        }

        // =========================================================
        // 1. Miss 逻辑：没打中物体，显示背景
        // =========================================================
        if (!scene.hit(r, 0.001, infinity, rec)) {
            for (const auto& light : lights) {
                if (!light->is_delta()) {
                    color env_color = light->Le(r);
                    if (env_color.length_squared() > 0) {
                        return env_color;
                    }
                }
            }
            return background;
        }

        color emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
        color L_total = emitted;

        ray scattered;
        color attenuation;

        // =========================================================
        // 2. 直接光照采样 (NEE)
        // =========================================================
        if (!lights.empty()) {
            for (const auto& light : lights) {
                // 采样光源
                LightSample ls = light->sample(rec.p, vec2(random_double(), random_double()));

                // 偏移起点防自遮挡
                vec3 offset_origin = rec.p + rec.normal * 1e-4;
                if (dot(rec.normal, ls.wi) < 0) {
                    offset_origin = rec.p - rec.normal * 1e-4;
                }

                // 发射阴影光线
                ray shadow_ray(offset_origin, ls.wi, r.time());
                hit_record shadow_rec;

                // ★★★ 修正后的阴影判定逻辑 ★★★
                // 如果没有打中障碍物 (注意 t_min=0, t_max=dist-epsilon)
                if (!scene.hit(shadow_ray, 0.0, ls.dist - 1e-3, shadow_rec)) {

                    // 可见！计算光照贡献
                    vec3 wo = -r.direction();
                    color f = rec.mat_ptr->eval(rec, wo, ls.wi);
                    double cos_theta = dot(rec.normal, ls.wi);

                    if (cos_theta > 0 && ls.pdf > 0) {
                        L_total += f * ls.Li * cos_theta / ls.pdf;
                    }
                }
            }
        }

        // =========================================================
        // 3. 间接光照
        // =========================================================
        if (rec.mat_ptr->scatter(r, rec, attenuation, scattered)) {
            L_total += attenuation * Li_internal(scattered, scene, lights, background, depth - 1);
        }

        return L_total;
    }

};

#endif
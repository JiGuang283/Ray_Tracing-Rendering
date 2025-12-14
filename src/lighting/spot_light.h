#ifndef SPOT_LIGHT_H
#define SPOT_LIGHT_H

#include "light.h"

class SpotLight : public Light {
public:
    // pos: 位置, dir: 照射方向
    // cutoff: 截止角度 (度), intensity: 强度
    SpotLight(point3 pos, vec3 dir, double cutoff, color intensity)
        : position(pos), direction(unit_vector(dir)), intensity(intensity) {
        // 将角度转换为余弦值，方便计算
        auto r = cutoff * (pi / 180.0);
        cos_cutoff = cos(r);
    }

    virtual LightSample sample(const point3& p, const vec2& u) const override {
        LightSample s;
        vec3 d = position - p;
        double dist2 = d.length_squared();
        s.dist = sqrt(dist2);
        s.wi = d / s.dist; // 指向光源
        s.is_delta = true;
        s.pdf = 1.0;

        // 计算聚光灯因子
        // 检查 光源射出的方向(-s.wi) 与 聚光灯朝向(direction) 的夹角
        double cos_theta = dot(-s.wi, direction);

        if (cos_theta < cos_cutoff) {
            // 在聚光灯范围外，无光
            s.Li = color(0, 0, 0);
        }
        else {
            // 在范围内 (简单起见，不加边缘柔化 Falloff)
            s.Li = intensity / dist2;
        }
        return s;
    }

    virtual bool is_delta() const override { return true; }

private:
    point3 position;
    vec3 direction;
    color intensity;
    double cos_cutoff;
};
#endif

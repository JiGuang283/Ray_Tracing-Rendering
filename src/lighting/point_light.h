// lighting/point_light.h
#ifndef POINT_LIGHT_H
#define POINT_LIGHT_H

#include "light.h"

class PointLight : public Light {
public:
    // position: 光源位置
    // intensity: 光强 (对于点光源，单位通常是 Watts/sr 或者简单的 RGB 强度)
    PointLight(const point3& position, const color& intensity)
        : position(position), intensity(intensity) {
    }

    virtual LightSample sample(const point3& p, const vec2& u) const override {
        LightSample s;
        vec3 direction = position - p;
        double dist_squared = direction.length_squared();

        s.dist = std::sqrt(dist_squared);
        s.wi = direction / s.dist; // 归一化方向
        s.is_delta = true;         // 点光源是 Delta 光源
        s.pdf = 1.0;               // Delta 光源 PDF 恒为 1 (而在立体角空间通常视为狄拉克分布处理)

        // 辐射衰减：光强 / 距离平方
        s.Li = intensity / dist_squared;

        return s;
    }

    virtual bool is_delta() const override {
        return true;
    }

private:
    point3 position;
    color intensity;
};

#endif
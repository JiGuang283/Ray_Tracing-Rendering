// lighting/directional_light.h
#ifndef DIRECTIONAL_LIGHT_H
#define DIRECTIONAL_LIGHT_H

#include "light.h"

class DirectionalLight : public Light {
public:
    // dir: 光线射过来的方向 (例如从正上方射下来是 (0, -1, 0))
    // c: 光的颜色/强度 (不随距离衰减)
    DirectionalLight(const vec3& dir, const color& c)
        : direction(unit_vector(dir)), L(c) {
    }

    virtual LightSample sample(const point3& p, const vec2& u) const override {
        LightSample s;
        // 平行光方向是固定的，指向光源的方向就是光线方向的反方向
        s.wi = -direction;
        s.dist = infinity; // 距离无限远
        s.Li = L;          // 强度不衰减
        s.is_delta = true;
        s.pdf = 1.0;
        return s;
    }

    virtual bool is_delta() const override {
        return true;
    }

private:
    vec3 direction;
    color L;
};

#endif
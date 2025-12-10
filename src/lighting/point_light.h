#ifndef POINT_LIGHT_H
#define POINT_LIGHT_H

#include "light.h"

class PointLight : public Light {
  public:
    PointLight(const point3 &pos, const color &intensity)
        : m_position(pos), m_intensity(intensity) {
    }

    virtual LightSample sample(const point3 &p, const vec2 &u) const override {
        LightSample ls;

        vec3 direction = m_position - p;
        double dist_squared = direction.length_squared();

        ls.dist = sqrt(dist_squared);
        ls.wi = direction / ls.dist;
        ls.Li = m_intensity / dist_squared;
        ls.pdf = 1.0;
        ls.is_delta = true;

        return ls;
    }

    virtual bool is_delta() const override {
        return true;
    }

    virtual color power() const override {
        return 4.0 * pi * m_intensity;
    }

  private:
    point3 m_position;
    color m_intensity;
};

#endif
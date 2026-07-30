#ifndef TRANSFORMED_LIGHT_H
#define TRANSFORMED_LIGHT_H

#include "light.h"

class TranslateLight : public Light {
  public:
    TranslateLight(shared_ptr<Light> light, const vec3 &offset)
        : m_light(std::move(light)), m_offset(offset) {
    }

    LightSample sample(const point3 &p, const vec2 &u) const override {
        return m_light->sample(p - m_offset, u);
    }

    double pdf(const point3 &origin, const vec3 &direction) const override {
        return m_light->pdf(origin - m_offset, direction);
    }

    bool is_delta() const override {
        return m_light->is_delta();
    }

    bool is_infinite() const override {
        return m_light->is_infinite();
    }

    color Le(const ray &r) const override {
        return m_light->Le(ray(r.origin() - m_offset, r.direction(),
                               r.time()));
    }

    color power() const override {
        return m_light->power();
    }

  private:
    shared_ptr<Light> m_light;
    vec3 m_offset;
};

class RotateYLight : public Light {
  public:
    RotateYLight(shared_ptr<Light> light, double angle)
        : m_light(std::move(light)) {
        double radians = degrees_to_radians(angle);
        m_sin_theta = sin(radians);
        m_cos_theta = cos(radians);
    }

    LightSample sample(const point3 &p, const vec2 &u) const override {
        LightSample sample = m_light->sample(to_local(p), u);
        sample.wi = to_world_direction(sample.wi);
        return sample;
    }

    double pdf(const point3 &origin, const vec3 &direction) const override {
        return m_light->pdf(to_local(origin), to_local_direction(direction));
    }

    bool is_delta() const override {
        return m_light->is_delta();
    }

    bool is_infinite() const override {
        return m_light->is_infinite();
    }

    color Le(const ray &r) const override {
        return m_light->Le(
            ray(to_local(r.origin()), to_local_direction(r.direction()),
                r.time()));
    }

    color power() const override {
        return m_light->power();
    }

  private:
    point3 to_local(const point3 &p) const {
        point3 result = p;
        result[0] = m_cos_theta * p[0] - m_sin_theta * p[2];
        result[2] = m_sin_theta * p[0] + m_cos_theta * p[2];
        return result;
    }

    vec3 to_local_direction(const vec3 &v) const {
        return to_local(v);
    }

    vec3 to_world_direction(const vec3 &v) const {
        vec3 result = v;
        result[0] = m_cos_theta * v[0] + m_sin_theta * v[2];
        result[2] = -m_sin_theta * v[0] + m_cos_theta * v[2];
        return result;
    }

    shared_ptr<Light> m_light;
    double m_sin_theta = 0.0;
    double m_cos_theta = 1.0;
};

#endif

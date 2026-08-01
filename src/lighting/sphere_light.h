#ifndef SPHERE_LIGHT_H
#define SPHERE_LIGHT_H

#include "light.h"
#include "rtweekend.h"

#include <algorithm>

class SphereLight : public Light {
  public:
    SphereLight(const point3 &center, double radius, const color &intensity)
        : m_center(center), m_radius(radius), m_intensity(intensity) {
        m_area = 4.0 * pi * m_radius * m_radius;
    }

    LightSample sample(const point3 &p, const vec2 &u) const override {
        LightSample s;
        double z = 1.0 - 2.0 * u.x();
        double r = sqrt(std::max(0.0, 1.0 - z * z));
        double phi = 2.0 * pi * u.y();
        vec3 normal(r * cos(phi), r * sin(phi), z);
        point3 light_point = m_center + m_radius * normal;

        vec3 d = light_point - p;
        double dist_sq = d.length_squared();
        s.dist = sqrt(dist_sq);
        s.wi = d / s.dist;
        s.is_delta = false;

        double cos_theta = dot(-s.wi, normal);
        if (m_area <= 0.0 || cos_theta <= 0.0) {
            s.Li = color(0, 0, 0);
            s.pdf = 0.0;
            return s;
        }

        s.Li = m_intensity;
        s.pdf = dist_sq / (m_area * cos_theta);
        return s;
    }

    double pdf(const point3 &origin, const vec3 &direction) const override {
        vec3 oc = origin - m_center;
        double a = direction.length_squared();
        double half_b = dot(oc, direction);
        double c = oc.length_squared() - m_radius * m_radius;
        double discriminant = half_b * half_b - a * c;
        if (discriminant < 0.0 || m_area <= 0.0) {
            return 0.0;
        }

        double sqrtd = sqrt(discriminant);
        double t = (-half_b - sqrtd) / a;
        if (t < 0.001) {
            t = (-half_b + sqrtd) / a;
        }
        if (t < 0.001) {
            return 0.0;
        }

        point3 hit_point = origin + t * direction;
        vec3 normal = unit_vector(hit_point - m_center);
        double cos_theta = -dot(unit_vector(direction), normal);
        if (cos_theta <= 0.0) {
            return 0.0;
        }

        double dist_sq = t * t * direction.length_squared();
        return dist_sq / (m_area * cos_theta);
    }

    color power() const override {
        return pi * m_area * m_intensity;
    }

    bool is_bsdf_hittable() const override {
        return true;
    }

  private:
    point3 m_center;
    double m_radius = 1.0;
    double m_area = 0.0;
    color m_intensity;
};

#endif

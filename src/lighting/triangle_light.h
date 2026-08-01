#ifndef TRIANGLE_LIGHT_H
#define TRIANGLE_LIGHT_H

#include "light.h"

class TriangleLight : public Light {
  public:
    TriangleLight(const point3 &a, const point3 &b, const point3 &c,
                  const color &intensity)
        : v0(a), v1(b), v2(c), intensity(intensity) {
        edge1 = v1 - v0;
        edge2 = v2 - v0;
        vec3 n = cross(edge1, edge2);
        area = 0.5 * n.length();
        normal = unit_vector(n);
    }

    LightSample sample(const point3 &p, const vec2 &u) const override {
        LightSample s;
        double su0 = sqrt(u.x());
        double b0 = 1.0 - su0;
        double b1 = u.y() * su0;
        double b2 = 1.0 - b0 - b1;
        point3 light_point = b0 * v0 + b1 * v1 + b2 * v2;

        vec3 d = light_point - p;
        double dist_sq = d.length_squared();
        s.dist = sqrt(dist_sq);
        s.wi = d / s.dist;
        s.is_delta = false;

        double cos_theta = dot(-s.wi, normal);
        if (area <= 0.0 || cos_theta <= 0.0) {
            s.Li = color(0, 0, 0);
            s.pdf = 0.0;
            return s;
        }

        s.Li = intensity;
        s.pdf = dist_sq / (area * cos_theta);
        return s;
    }

    double pdf(const point3 &origin, const vec3 &direction) const override {
        const double eps = 1e-8;
        vec3 pvec = cross(direction, edge2);
        double det = dot(edge1, pvec);
        if (fabs(det) < eps) {
            return 0.0;
        }

        double inv_det = 1.0 / det;
        vec3 tvec = origin - v0;
        double u = dot(tvec, pvec) * inv_det;
        if (u < 0.0 || u > 1.0) {
            return 0.0;
        }

        vec3 qvec = cross(tvec, edge1);
        double v = dot(direction, qvec) * inv_det;
        if (v < 0.0 || u + v > 1.0) {
            return 0.0;
        }

        double t = dot(edge2, qvec) * inv_det;
        if (t < 0.001 || area <= 0.0) {
            return 0.0;
        }

        double dist_sq = t * t * direction.length_squared();
        double cos_theta = -dot(unit_vector(direction), normal);
        if (cos_theta <= 0.0) {
            return 0.0;
        }
        return dist_sq / (area * cos_theta);
    }

    color power() const override {
        return pi * area * intensity;
    }

    bool is_bsdf_hittable() const override {
        return true;
    }

  private:
    point3 v0;
    point3 v1;
    point3 v2;
    vec3 edge1;
    vec3 edge2;
    vec3 normal;
    double area = 0.0;
    color intensity;
};

#endif

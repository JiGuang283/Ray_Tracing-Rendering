#ifndef TRIANGLE_LIGHT_H
#define TRIANGLE_LIGHT_H

#include "light.h"
#include "material.h"
#include "material_emission.h"

#include <cmath>
#include <utility>

class TriangleLight : public Light {
  public:
    TriangleLight(const point3 &a, const point3 &b, const point3 &c,
                  const color &intensity)
        : v0(a), v1(b), v2(c), intensity(intensity) {
        initialize_geometry();
    }

    TriangleLight(const point3 &a, const point3 &b, const point3 &c,
                  MaterialHandle material, const vec2 &texture0 = vec2(0, 0),
                  const vec2 &texture1 = vec2(0, 0),
                  const vec2 &texture2 = vec2(0, 0), bool has_uvs = false)
        : v0(a), v1(b), v2(c), uv0(texture0), uv1(texture1), uv2(texture2),
          m_material(std::move(material)), m_has_uvs(has_uvs) {
        initialize_geometry();
        m_double_sided = m_material && m_material->is_double_sided();
        intensity = estimate_emission();
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
        if (area <= 0.0 || !std::isfinite(dist_sq) || dist_sq <= 1e-12) {
            return s;
        }
        s.dist = sqrt(dist_sq);
        s.wi = d / s.dist;
        s.is_delta = false;

        const double signed_cosine = dot(-s.wi, normal);
        const double cos_theta =
            m_double_sided ? std::abs(signed_cosine) : signed_cosine;
        if (cos_theta <= 0.0) {
            return s;
        }

        if (m_material) {
            const bool front_face = signed_cosine > 0.0;
            const vec3 oriented_normal = front_face ? normal : -normal;
            s.Li = evaluate_material_emission(
                m_material, light_point, texture_coordinates(b0, b1, b2),
                oriented_normal, dpdu, dpdv, front_face);
        } else {
            s.Li = intensity;
        }
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
        if (m_double_sided) {
            cos_theta = std::abs(cos_theta);
        }
        if (cos_theta <= 0.0) {
            return 0.0;
        }
        return dist_sq / (area * cos_theta);
    }

    color power() const override {
        return (m_double_sided ? 2.0 : 1.0) * pi * area * intensity;
    }

    bool is_bsdf_hittable() const override {
        return true;
    }

  private:
    void initialize_geometry() {
        edge1 = v1 - v0;
        edge2 = v2 - v0;
        const vec3 n = cross(edge1, edge2);
        const double twice_area = n.length();
        area = 0.5 * twice_area;
        normal = twice_area > 0.0 ? n / twice_area : vec3(0, 0, 1);
        dpdu = edge1;
        dpdv = edge2;
        if (m_has_uvs) {
            const vec2 duv1 = uv1 - uv0;
            const vec2 duv2 = uv2 - uv0;
            const double determinant =
                duv1.x() * duv2.y() - duv1.y() * duv2.x();
            if (std::abs(determinant) > 1e-10) {
                const double inverse = 1.0 / determinant;
                dpdu =
                    (duv2.y() * edge1 - duv1.y() * edge2) * inverse;
                dpdv =
                    (-duv2.x() * edge1 + duv1.x() * edge2) * inverse;
            }
        }
    }

    vec2 texture_coordinates(double b0, double b1, double b2) const {
        return m_has_uvs ? b0 * uv0 + b1 * uv1 + b2 * uv2
                         : vec2(b1, b2);
    }

    color estimate_emission() const {
        if (!m_material || area <= 0.0) {
            return color(0, 0, 0);
        }
        constexpr double barycentrics[4][3] = {
            {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0},
            {0.6, 0.2, 0.2},
            {0.2, 0.6, 0.2},
            {0.2, 0.2, 0.6}};
        color estimate(0, 0, 0);
        for (const auto &b : barycentrics) {
            estimate += evaluate_material_emission(
                m_material, b[0] * v0 + b[1] * v1 + b[2] * v2,
                texture_coordinates(b[0], b[1], b[2]), normal, dpdu, dpdv,
                true);
        }
        return estimate / 4.0;
    }

    point3 v0;
    point3 v1;
    point3 v2;
    vec3 edge1;
    vec3 edge2;
    vec3 normal;
    vec3 dpdu;
    vec3 dpdv;
    vec2 uv0{0, 0};
    vec2 uv1{0, 0};
    vec2 uv2{0, 0};
    double area = 0.0;
    color intensity;
    MaterialHandle m_material;
    bool m_has_uvs = false;
    bool m_double_sided = false;
};

#endif

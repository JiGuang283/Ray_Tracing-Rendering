#ifndef QUAD_LIGHT_H
#define QUAD_LIGHT_H

#include "light.h"
#include "material.h"
#include "material_emission.h"
#include "rtweekend.h"

#include <cmath>
#include <utility>

class QuadLight : public Light {
  public:
    QuadLight(const point3 &_Q, const vec3 &_u, const vec3 &_v,
              const color &_c, bool bsdf_hittable = false)
        : Q(_Q), u(_u), v(_v), intensity(_c),
          m_bsdf_hittable(bsdf_hittable) {
        initialize_geometry();
    }

    QuadLight(const point3 &_Q, const vec3 &_u, const vec3 &_v,
              MaterialHandle material, bool bsdf_hittable = true,
              bool swap_uv = false)
        : Q(_Q), u(_u), v(_v), m_material(std::move(material)),
          m_swap_uv(swap_uv), m_bsdf_hittable(bsdf_hittable) {
        initialize_geometry();
        m_double_sided = m_material && m_material->is_double_sided();
        intensity = estimate_emission();
    }

    virtual LightSample sample(const point3 &p,
                               const vec2 &random_u) const override {
        LightSample s;
        // Sample point on the rectangle
        point3 light_point = Q + random_u.x() * u + random_u.y() * v;

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

        if (cos_theta <= 0) {
            return s;
        }

        if (m_material) {
            const bool front_face = signed_cosine > 0.0;
            const vec3 oriented_normal = front_face ? normal : -normal;
            s.Li = evaluate_material_emission(
                m_material, light_point, texture_uv(random_u),
                oriented_normal, u, v, front_face);
        } else {
            s.Li = intensity;
        }

        // Convert area PDF to solid angle PDF
        // pdf_area = 1.0 / area
        // pdf_solid = pdf_area * dist^2 / cos_theta
        s.pdf = dist_sq / (area * cos_theta);

        return s;
    }

    virtual double pdf(const point3 &origin,
                       const vec3 &direction) const override {
        // Ray-plane intersection
        double denom = dot(direction, normal);

        if ((!m_double_sided && denom >= -1e-6) ||
            (m_double_sided && std::abs(denom) <= 1e-6))
            return 0;

        double t = dot(Q - origin, normal) / denom;
        if (t < 0.001 || t > infinity)
            return 0;

        point3 intersection = origin + t * direction;
        vec3 planar_hit_pt_vector = intersection - Q;

        // Assuming orthogonal u, v
        double alpha = dot(planar_hit_pt_vector, u) / u.length_squared();
        double beta = dot(planar_hit_pt_vector, v) / v.length_squared();

        if (alpha < 0 || alpha > 1 || beta < 0 || beta > 1) {
            return 0;
        }

        double dist_sq = t * t * direction.length_squared();
        double cos_theta = -denom / direction.length();
        if (m_double_sided) {
            cos_theta = std::abs(cos_theta);
        }

        return dist_sq / (area * cos_theta);
    }

    virtual bool is_delta() const override {
        return false;
    }
    virtual bool is_infinite() const override {
        return false;
    }

    bool is_bsdf_hittable() const override {
        return m_bsdf_hittable;
    }

    virtual color power() const override {
        return (m_double_sided ? 2.0 : 1.0) * pi * area * intensity;
    }

  private:
    void initialize_geometry() {
        const vec3 n = cross(u, v);
        area = n.length();
        normal = area > 0.0 ? n / area : vec3(0, 0, 1);
    }

    color estimate_emission() const {
        if (!m_material || area <= 0.0) {
            return color(0, 0, 0);
        }
        constexpr double samples[4][2] = {
            {0.25, 0.25}, {0.75, 0.25}, {0.25, 0.75}, {0.75, 0.75}};
        color estimate(0, 0, 0);
        for (const auto &sample : samples) {
            const vec2 uv(sample[0], sample[1]);
            estimate += evaluate_material_emission(
                m_material, Q + uv.x() * u + uv.y() * v, texture_uv(uv),
                normal, u, v, true);
        }
        return estimate / 4.0;
    }

    vec2 texture_uv(const vec2 &uv) const {
        return m_swap_uv ? vec2(uv.y(), uv.x()) : uv;
    }

    point3 Q;
    vec3 u, v;
    color intensity;
    vec3 normal;
    double area = 0.0;
    MaterialHandle m_material;
    bool m_double_sided = false;
    bool m_swap_uv = false;
    bool m_bsdf_hittable = false;
};

#endif

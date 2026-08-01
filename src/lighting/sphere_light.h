#ifndef SPHERE_LIGHT_H
#define SPHERE_LIGHT_H

#include "light.h"
#include "material.h"
#include "material_emission.h"
#include "rtweekend.h"

#include <algorithm>
#include <cmath>
#include <utility>

class SphereLight : public Light {
  public:
    SphereLight(const point3 &center, double radius, const color &intensity)
        : m_center(center), m_radius(radius), m_intensity(intensity) {
        m_area = 4.0 * pi * m_radius * m_radius;
    }

    SphereLight(const point3 &center, double radius, MaterialHandle material)
        : m_center(center), m_radius(radius),
          m_material(std::move(material)) {
        m_area = 4.0 * pi * m_radius * m_radius;
        m_double_sided = m_material && m_material->is_double_sided();
        m_intensity = estimate_emission();
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
        if (m_area <= 0.0 || !std::isfinite(dist_sq) ||
            dist_sq <= 1e-12) {
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
            double texture_u = 0.0;
            double texture_v = 0.0;
            sphere_uv(normal, texture_u, texture_v);
            const bool front_face = signed_cosine > 0.0;
            const vec3 oriented_normal = front_face ? normal : -normal;
            vec3 dpdu;
            vec3 dpdv;
            sphere_derivatives(normal, dpdu, dpdv);
            s.Li = evaluate_material_emission(
                m_material, light_point, vec2(texture_u, texture_v),
                oriented_normal, dpdu, dpdv, front_face);
        } else {
            s.Li = m_intensity;
        }
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
        if (m_double_sided) {
            cos_theta = std::abs(cos_theta);
        }
        if (cos_theta <= 0.0) {
            return 0.0;
        }

        double dist_sq = t * t * direction.length_squared();
        return dist_sq / (m_area * cos_theta);
    }

    color power() const override {
        return (m_double_sided ? 2.0 : 1.0) * pi * m_area * m_intensity;
    }

    bool is_bsdf_hittable() const override {
        return true;
    }

  private:
    static void sphere_uv(const vec3 &normal, double &u, double &v) {
        const double theta = acos(clamp(-normal.y(), -1.0, 1.0));
        const double phi = atan2(-normal.z(), normal.x()) + pi;
        u = phi / (2.0 * pi);
        v = theta / pi;
    }

    void sphere_derivatives(const vec3 &normal, vec3 &dpdu,
                            vec3 &dpdv) const {
        double u = 0.0;
        double v = 0.0;
        sphere_uv(normal, u, v);
        const double theta = v * pi;
        const double alpha = u * 2.0 * pi - pi;
        dpdu = 2.0 * pi * m_radius *
               vec3(-sin(alpha) * sin(theta), 0.0,
                    -cos(alpha) * sin(theta));
        dpdv = pi * m_radius *
               vec3(cos(alpha) * cos(theta), sin(theta),
                    -sin(alpha) * cos(theta));
    }

    color estimate_emission() const {
        if (!m_material || m_area <= 0.0) {
            return color(0, 0, 0);
        }
        const vec3 normals[] = {vec3(1, 0, 0),  vec3(-1, 0, 0),
                                vec3(0, 1, 0),  vec3(0, -1, 0),
                                vec3(0, 0, 1),  vec3(0, 0, -1)};
        color estimate(0, 0, 0);
        for (const vec3 &normal : normals) {
            double u = 0.0;
            double v = 0.0;
            vec3 dpdu;
            vec3 dpdv;
            sphere_uv(normal, u, v);
            sphere_derivatives(normal, dpdu, dpdv);
            estimate += evaluate_material_emission(
                m_material, m_center + m_radius * normal, vec2(u, v), normal,
                dpdu, dpdv, true);
        }
        return estimate / 6.0;
    }

    point3 m_center;
    double m_radius = 1.0;
    double m_area = 0.0;
    color m_intensity;
    MaterialHandle m_material;
    bool m_double_sided = false;
};

#endif

#ifndef MESH_LIGHT_H
#define MESH_LIGHT_H

#include "light.h"
#include "triangle_surface.h"

#include <algorithm>
#include <vector>

class MeshLight : public Light {
  public:
    MeshLight(const std::vector<TriangleSurface> &triangles,
              const color &intensity)
        : m_intensity(intensity) {
        m_triangles.reserve(triangles.size());
        m_cdf.reserve(triangles.size());
        double accum = 0.0;
        for (const auto &triangle : triangles) {
            TriangleEntry entry;
            entry.v0 = triangle.v0;
            entry.v1 = triangle.v1;
            entry.v2 = triangle.v2;
            entry.edge1 = entry.v1 - entry.v0;
            entry.edge2 = entry.v2 - entry.v0;
            vec3 n = cross(entry.edge1, entry.edge2);
            entry.area = 0.5 * n.length();
            if (entry.area <= 0.0) {
                continue;
            }
            entry.normal = unit_vector(n);
            accum += entry.area;
            m_triangles.push_back(entry);
            m_cdf.push_back(accum);
        }
        m_total_area = accum;
    }

    LightSample sample(const point3 &p, const vec2 &u) const override {
        LightSample s;
        s.Li = color(0, 0, 0);
        s.wi = vec3(0, 0, 1);
        s.pdf = 0.0;
        s.dist = infinity;
        s.is_delta = false;

        if (m_triangles.empty() || m_total_area <= 0.0) {
            return s;
        }

        double local_u = 0.0;
        const TriangleEntry &tri = choose_triangle(u.x(), local_u);
        double su0 = sqrt(local_u);
        double b0 = 1.0 - su0;
        double b1 = u.y() * su0;
        double b2 = 1.0 - b0 - b1;
        point3 light_point = b0 * tri.v0 + b1 * tri.v1 + b2 * tri.v2;

        vec3 d = light_point - p;
        double dist_sq = d.length_squared();
        s.dist = sqrt(dist_sq);
        s.wi = d / s.dist;

        double cos_theta = dot(-s.wi, tri.normal);
        if (cos_theta <= 0.0) {
            return s;
        }

        s.Li = m_intensity;
        s.pdf = dist_sq / (m_total_area * cos_theta);
        return s;
    }

    double pdf(const point3 &origin, const vec3 &direction) const override {
        if (m_triangles.empty() || m_total_area <= 0.0) {
            return 0.0;
        }

        double closest_t = infinity;
        double selected_cos = 0.0;
        for (const auto &tri : m_triangles) {
            double t = 0.0;
            if (intersect_triangle(tri, origin, direction, t) &&
                t < closest_t) {
                closest_t = t;
                selected_cos = -dot(unit_vector(direction), tri.normal);
            }
        }
        if (closest_t == infinity || selected_cos <= 0.0) {
            return 0.0;
        }

        double dist_sq = closest_t * closest_t * direction.length_squared();
        return dist_sq / (m_total_area * selected_cos);
    }

    color power() const override {
        return pi * m_total_area * m_intensity;
    }

  private:
    struct TriangleEntry {
        point3 v0;
        point3 v1;
        point3 v2;
        vec3 edge1;
        vec3 edge2;
        vec3 normal;
        double area = 0.0;
    };

    const TriangleEntry &choose_triangle(double u, double &local_u) const {
        double target = clamp(u, 0.0, 0.999999999) * m_total_area;
        auto found = std::lower_bound(m_cdf.begin(), m_cdf.end(), target);
        int index = static_cast<int>(found - m_cdf.begin());
        index = std::min(index, static_cast<int>(m_triangles.size()) - 1);

        double previous = index == 0 ? 0.0 : m_cdf[index - 1];
        double width = m_cdf[index] - previous;
        local_u = width > 0.0 ? (target - previous) / width : 0.0;
        local_u = clamp(local_u, 0.0, 0.999999999);
        return m_triangles[index];
    }

    static bool intersect_triangle(const TriangleEntry &tri,
                                   const point3 &origin,
                                   const vec3 &direction, double &t) {
        const double eps = 1e-8;
        vec3 pvec = cross(direction, tri.edge2);
        double det = dot(tri.edge1, pvec);
        if (fabs(det) < eps) {
            return false;
        }
        double inv_det = 1.0 / det;
        vec3 tvec = origin - tri.v0;
        double u = dot(tvec, pvec) * inv_det;
        if (u < 0.0 || u > 1.0) {
            return false;
        }
        vec3 qvec = cross(tvec, tri.edge1);
        double v = dot(direction, qvec) * inv_det;
        if (v < 0.0 || u + v > 1.0) {
            return false;
        }
        t = dot(tri.edge2, qvec) * inv_det;
        return t >= 0.001;
    }

    std::vector<TriangleEntry> m_triangles;
    std::vector<double> m_cdf;
    double m_total_area = 0.0;
    color m_intensity;
};

#endif

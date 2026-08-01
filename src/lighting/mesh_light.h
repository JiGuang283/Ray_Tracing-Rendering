#ifndef MESH_LIGHT_H
#define MESH_LIGHT_H

#include "light.h"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

class MeshInstance;

class MeshLight final : public Light {
  public:
    MeshLight(std::shared_ptr<const MeshInstance> instance,
              std::uint32_t material_slot, bool flip_orientation = false);

    LightSample sample(const point3 &p, const vec2 &u) const override;
    double pdf(const point3 &origin,
               const vec3 &direction) const override;
    color power() const override;
    bool is_bsdf_hittable() const override;

  private:
    struct TriangleEntry {
        std::uint32_t triangle_index = 0;
        std::array<std::uint8_t, 3> corner_order{0, 1, 2};
        point3 v0;
        point3 v1;
        point3 v2;
        vec3 edge1;
        vec3 edge2;
        vec3 normal;
        double area = 0.0;
    };

    const TriangleEntry &choose_triangle(double u, double &local_u) const;
    color evaluate_emission(const TriangleEntry &triangle, double b0,
                            double b1, double b2, bool front_face) const;
    static bool intersect_triangle(const TriangleEntry &triangle,
                                   const point3 &origin,
                                   const vec3 &direction, double &t);

    std::shared_ptr<const MeshInstance> m_instance;
    std::uint32_t m_material_slot = 0;
    std::vector<TriangleEntry> m_triangles;
    std::vector<double> m_cdf;
    double m_total_area = 0.0;
    bool m_double_sided = false;
};

#endif

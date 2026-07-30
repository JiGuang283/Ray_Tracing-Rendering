#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "material.h"
#include "vec3.h"

class sphere : public hittable {
  public:
    sphere(point3 cen, double r, MaterialHandle m)
        : center(cen), radius(r), mat_ptr(std::move(m)) {};

    virtual bool hit(const ray &r, double t_min, double t_max,
                     hit_record &rec) const override;

    virtual bool bounding_box(double time0, double time1,
                              aabb &output_box) const override;

  public:
    point3 center;
    double radius;
    MaterialHandle mat_ptr;

  private:
    static void get_sphere_uv(const point3 &p, double &u, double &v);
};

#endif

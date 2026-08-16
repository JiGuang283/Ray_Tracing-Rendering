#ifndef BOX_H
#define BOX_H

#include "aarect.h"
#include "hittable_list.h"
#include "ray.h"
#include "rtweekend.h"
#include "vec3.h"

class box : public hittable {
  public:
    box() {
    }
    box(const point3 &p0, const point3 &p1, MaterialHandle ptr);

    virtual bool hit(const ray &r, double t_min, double t_max,
                     hit_record &rec) const override;
    virtual bool hit(const ray &r, double t_min, double t_max,
                     hit_record &rec, RNG &rng) const override;
    virtual bool occluded(const ray &r, double t_min, double t_max,
                          RNG &rng) const override;

    virtual bool bounding_box(double /*time0*/, double /*time1*/, 
                              aabb &output_box) const override {
        output_box = aabb(box_min, box_max);
        return true;
    }

  public:
    point3 box_min;
    point3 box_max;
    hittable_list sides;
};

#endif

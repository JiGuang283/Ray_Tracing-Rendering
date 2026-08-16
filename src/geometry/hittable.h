#ifndef HITTABLE_H
#define HITTABLE_H

#include "aabb.h"
#include "interaction.h"
#include "ray.h"
#include "rtweekend.h"

class hittable {
  public:
    virtual ~hittable() = default;
    virtual bool hit(const ray &r, double t_min, double t_max,
                     hit_record &rec) const = 0;
    virtual bool hit(const ray &r, double t_min, double t_max,
                     hit_record &rec, RNG & /*rng*/) const {
        return hit(r, t_min, t_max, rec);
    }
    virtual bool occluded(const ray &r, double t_min, double t_max,
                          RNG &rng) const {
        hit_record rec;
        return hit(r, t_min, t_max, rec, rng);
    }
    virtual bool bounding_box(double time0, double time1,
                              aabb &output_box) const = 0;
};

class flip_face : public hittable {
  public:
    flip_face(shared_ptr<hittable> p) : ptr(p) {}

    virtual bool hit(const ray& r, double t_min, double t_max,
                     hit_record& rec) const override {
        RNG rng(make_thread_seed());
        return hit(r, t_min, t_max, rec, rng);
    }

    virtual bool hit(const ray& r, double t_min, double t_max,
                     hit_record& rec, RNG &rng) const override {
        if (!ptr->hit(r, t_min, t_max, rec, rng)) return false;
        rec.front_face = !rec.front_face;
        rec.normal = -rec.normal;
        rec.geometric_normal = -rec.geometric_normal;
        return true;
    }

    virtual bool occluded(const ray& r, double t_min, double t_max,
                          RNG &rng) const override {
        return ptr->occluded(r, t_min, t_max, rng);
    }

    virtual bool bounding_box(double time0, double time1,
                              aabb& output_box) const override {
        return ptr->bounding_box(time0, time1, output_box);
    }

  public:
    shared_ptr<hittable> ptr;
};



#endif

#ifndef CONSTANT_MEDIUM_H
#define CONSTANT_MEDIUM_H

#include "rtweekend.h"

#include "hittable.h"
#include "material.h"
#include "material_programs.h"
#include "ray.h"
#include "texture.h"
#include "vec3.h"

class constant_medium : public hittable {
  public:
    constant_medium(shared_ptr<hittable> b, double d, TextureHandle a)
        : boundary(b), phase_function(make_isotropic_material(std::move(a))),
          neg_inv_density(-1 / d) {
    }

    constant_medium(shared_ptr<hittable> b, double d, color c)
        : boundary(b), phase_function(make_isotropic_material(c)),
          neg_inv_density(-1 / d) {
    }

    virtual bool hit(const ray &r, double t_min, double t_max,
                     hit_record &rec) const override;
    virtual bool hit(const ray &r, double t_min, double t_max,
                     hit_record &rec, RNG &rng) const override;
    virtual bool bounding_box(double time0, double time1,
                              aabb &output_box) const override {
        return boundary->bounding_box(time0, time1, output_box);
    }

  public:
    shared_ptr<hittable> boundary;
    MaterialHandle phase_function;
    double neg_inv_density;
};

inline bool constant_medium::hit(const ray &r, double t_min, double t_max,
                                 hit_record &rec) const {
    RNG rng(make_thread_seed());
    return hit(r, t_min, t_max, rec, rng);
}

inline bool constant_medium::hit(const ray &r, double t_min, double t_max,
                                 hit_record &rec, RNG &rng) const {
    // Print occasional samples when debugging. To enable, set enableDebug true.
    const bool enableDebug = false;
    const bool debugging = enableDebug && rng.next() < 0.00001;

    hit_record rec1, rec2;

    if (!boundary->hit(r, -infinity, infinity, rec1))
        return false;

    if (!boundary->hit(r, rec1.t + 0.0001, infinity, rec2))
        return false;

    if (debugging)
        std::cerr << "\nt_min=" << rec1.t << ", t_max=" << rec2.t << '\n';

    if (rec1.t < t_min)
        rec1.t = t_min;
    if (rec2.t > t_max)
        rec2.t = t_max;

    if (rec1.t >= rec2.t)
        return false;

    if (rec1.t < 0)
        rec1.t = 0;

    const auto ray_length = r.direction().length();
    const auto distance_inside_boundary = (rec2.t - rec1.t) * ray_length;
    const auto hit_distance = neg_inv_density * log(rng.next());

    if (hit_distance > distance_inside_boundary)
        return false;

    rec.t = rec1.t + hit_distance / ray_length;
    rec.p = r.at(rec.t);

    if (debugging) {
        std::cerr << "hit_distance = " << hit_distance << '\n'
                  << "rec.t = " << rec.t << '\n'
                  << "rec.p = " << rec.p << '\n';
    }

    rec.normal = vec3(1, 0, 0); // arbitrary
    rec.geometric_normal = rec.normal;
    rec.dpdu = vec3(0, 1, 0);
    rec.dpdv = vec3(0, 0, 1);
    rec.front_face = true;      // also arbitrary
    rec.mat_ptr = phase_function.get();

    return true;
}

#endif

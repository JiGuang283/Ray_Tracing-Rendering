#ifndef CONSTANT_MEDIUM_H
#define CONSTANT_MEDIUM_H

#include "rtweekend.h"

#include "hittable.h"
#include "material.h"
#include "ray.h"
#include "texture.h"
#include "vec3.h"

class isotropic : public material {
  public:
    isotropic(color c) : albedo(make_shared<solid_color>(c)) {
    }
    isotropic(shared_ptr<texture> a) : albedo(a) {
    }

    virtual bool sample(const hit_record &rec, const vec3 &wo,
                        BSDFSample &sampled) const override {
        RNG rng(make_thread_seed());
        return sample(rec, wo, sampled, rng);
    }

    virtual bool sample(const hit_record &rec, const vec3 &wo,
                        BSDFSample &sampled, RNG &rng) const override {
        sampled.wi = random_unit_vector(rng);
        sampled.f = albedo->value(rec.u, rec.v, rec.p) / (4.0 * pi);
        sampled.pdf = 1.0 / (4.0 * pi);
        sampled.is_specular = false;
        sampled.is_transmission = false;
        return true;
    }

    virtual color eval(const hit_record &rec, const vec3 &wo,
                       const vec3 &wi) const override {
        return albedo->value(rec.u, rec.v, rec.p) / (4.0 * pi);
    }

    virtual double pdf(const hit_record &rec, const vec3 &wo,
                       const vec3 &wi) const override {
        return 1.0 / (4.0 * pi);
    }

    virtual bool scatter(const ray &r_in, const hit_record &rec,
                         color &attenuation, ray &scattered) const override {
        RNG rng(make_thread_seed());
        return scatter(r_in, rec, attenuation, scattered, rng);
    }

    virtual bool scatter(const ray &r_in, const hit_record &rec,
                         color &attenuation, ray &scattered,
                         RNG &rng) const override {
        scattered = ray(rec.p, random_in_unit_sphere(rng), r_in.time());
        attenuation = albedo->value(rec.u, rec.v, rec.p);
        return true;
    }

  public:
    shared_ptr<texture> albedo;
};

class constant_medium : public hittable {
  public:
    constant_medium(shared_ptr<hittable> b, double d, shared_ptr<texture> a)
        : boundary(b), neg_inv_density(-1 / d),
          phase_function(make_shared<isotropic>(a)) {
    }

    constant_medium(shared_ptr<hittable> b, double d, color c)
        : boundary(b), neg_inv_density(-1 / d),
          phase_function(make_shared<isotropic>(c)) {
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
    shared_ptr<material> phase_function;
    double neg_inv_density;
};

bool constant_medium::hit(const ray &r, double t_min, double t_max,
                          hit_record &rec) const {
    RNG rng(make_thread_seed());
    return hit(r, t_min, t_max, rec, rng);
}

bool constant_medium::hit(const ray &r, double t_min, double t_max,
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
    rec.front_face = true;      // also arbitrary
    rec.mat_ptr = phase_function.get();

    return true;
}

#endif

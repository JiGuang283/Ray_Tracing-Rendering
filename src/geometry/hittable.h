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
                     hit_record &rec, RNG &rng) const {
        return hit(r, t_min, t_max, rec);
    }
    virtual bool bounding_box(double time0, double time1,
                              aabb &output_box) const = 0;
};

class translate : public hittable {
  public:
    translate(shared_ptr<hittable> p, const vec3 &displacement)
        : ptr(p), offset(displacement) {
    }

    virtual bool hit(const ray &r, double t_min, double t_max,
                     hit_record &rec) const override;
    virtual bool hit(const ray &r, double t_min, double t_max,
                     hit_record &rec, RNG &rng) const override;

    virtual bool bounding_box(double time0, double time1,
                              aabb &output_box) const override;

  public:
    shared_ptr<hittable> ptr;
    vec3 offset;
};

inline bool translate::hit(const ray &r, double t_min, double t_max,
                           hit_record &rec) const {
    RNG rng(make_thread_seed());
    return hit(r, t_min, t_max, rec, rng);
}

inline bool translate::hit(const ray &r, double t_min, double t_max,
                           hit_record &rec, RNG &rng) const {
    ray moved_r(r.origin() - offset, r.direction(), r.time());
    if (!ptr->hit(moved_r, t_min, t_max, rec, rng)) {
        return false;
    }

    rec.p += offset;

    return true;
}

inline bool translate::bounding_box(double time0, double time1,
                                    aabb &output_box) const {
    if (!ptr->bounding_box(time0, time1, output_box)) {
        return false;
    }

    output_box = aabb(output_box.min() + offset, output_box.max() + offset);

    return true;
}

class rotate_y : public hittable {
  public:
    rotate_y(shared_ptr<hittable> p, double angle);

    virtual bool hit(const ray &r, double t_min, double t_max,
                     hit_record &rec) const override;
    virtual bool hit(const ray &r, double t_min, double t_max,
                     hit_record &rec, RNG &rng) const override;

    virtual bool bounding_box(double time0, double time1,
                              aabb &output_box) const override {
        output_box = bbox;
        return hasbox;
    }

  public:
    shared_ptr<hittable> ptr;
    double sin_theta;
    double cos_theta;
    bool hasbox;
    aabb bbox;
};

inline rotate_y::rotate_y(shared_ptr<hittable> p, double angle) : ptr(p) {
    auto radians = degrees_to_radians(angle);
    sin_theta = sin(radians);
    cos_theta = cos(radians);
    hasbox = ptr->bounding_box(0, 1, bbox);

    point3 min(infinity, infinity, infinity);
    point3 max(-infinity, -infinity, -infinity);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                auto x = i * bbox.max().x() + (1 - i) * bbox.min().x();
                auto y = j * bbox.max().y() + (1 - j) * bbox.min().y();
                auto z = k * bbox.max().z() + (1 - k) * bbox.min().z();

                auto newx = cos_theta * x + sin_theta * z;
                auto newz = -sin_theta * x + cos_theta * z;

                vec3 tester(newx, y, newz);

                for (int c = 0; c < 3; c++) {
                    min[c] = fmin(min[c], tester[c]);
                    max[c] = fmax(max[c], tester[c]);
                }
            }
        }
    }
    bbox = aabb(min, max);
}

inline bool rotate_y::hit(const ray &r, double t_min, double t_max,
                          hit_record &rec) const {
    RNG rng(make_thread_seed());
    return hit(r, t_min, t_max, rec, rng);
}

inline bool rotate_y::hit(const ray &r, double t_min, double t_max,
                          hit_record &rec, RNG &rng) const {
    auto origin = r.origin();
    auto direction = r.direction();

    origin[0] = cos_theta * r.origin()[0] - sin_theta * r.origin()[2];
    origin[2] = sin_theta * r.origin()[0] + cos_theta * r.origin()[2];

    direction[0] = cos_theta * r.direction()[0] - sin_theta * r.direction()[2];
    direction[2] = sin_theta * r.direction()[0] + cos_theta * r.direction()[2];

    ray rotated_r(origin, direction, r.time());

    if (!ptr->hit(rotated_r, t_min, t_max, rec, rng))
        return false;

    auto rotate_to_world = [&](const vec3 &value) {
        vec3 rotated = value;
        rotated[0] = cos_theta * value[0] + sin_theta * value[2];
        rotated[2] = -sin_theta * value[0] + cos_theta * value[2];
        return rotated;
    };

    rec.p = rotate_to_world(rec.p);
    rec.normal = rotate_to_world(rec.normal);
    rec.geometric_normal = rotate_to_world(rec.geometric_normal);
    rec.dpdu = rotate_to_world(rec.dpdu);
    rec.dpdv = rotate_to_world(rec.dpdv);
    return true;
}

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

    virtual bool bounding_box(double time0, double time1,
                              aabb& output_box) const override {
        return ptr->bounding_box(time0, time1, output_box);
    }

  public:
    shared_ptr<hittable> ptr;
};



#endif

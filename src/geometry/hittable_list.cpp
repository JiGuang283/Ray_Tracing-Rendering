#include "hittable_list.h"

bool hittable_list::hit(const ray &r, double t_min, double t_max,
                        hit_record &rec) const {
    RNG rng(make_thread_seed());
    return hit(r, t_min, t_max, rec, rng);
}

bool hittable_list::hit(const ray &r, double t_min, double t_max,
                        hit_record &rec, RNG &rng) const {
    hit_record temp_rec;
    bool hit_anything = false;
    auto closest_so_far = t_max;

    for (const auto &object : objects) {
        if (object->hit(r, t_min, closest_so_far, temp_rec, rng)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
}

bool hittable_list::occluded(const ray &r, double t_min, double t_max,
                              RNG &rng) const {
    for (const auto &object : objects) {
        if (object->occluded(r, t_min, t_max, rng)) {
            return true;
        }
    }
    return false;
}

bool hittable_list::bounding_box(double time0, double time1,
                                 aabb &output_box) const {
    if (objects.empty()) {
        return false;
    }

    aabb temp_box;
    bool first_box = true;

    for (const auto &object : objects) {
        if (!object->bounding_box(time0, time1, temp_box)) {
            return false;
        }
        output_box =
            first_box ? temp_box : surrounding_box(output_box, temp_box);
        first_box = false;
    }
    return true;
}

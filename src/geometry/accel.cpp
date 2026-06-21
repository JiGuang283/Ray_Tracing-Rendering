#include "accel.h"

#include "bvh.h"

shared_ptr<hittable> make_accel(const hittable_list &world, double time0,
                                double time1) {
    return make_shared<LinearBVH>(world, time0, time1);
}

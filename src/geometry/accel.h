#ifndef ACCEL_H
#define ACCEL_H

#include "hittable.h"
#include "hittable_list.h"

shared_ptr<hittable> make_accel(const hittable_list &world,
                                double time0 = 0.0,
                                double time1 = 1.0);

#endif

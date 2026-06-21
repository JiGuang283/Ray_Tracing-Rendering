#ifndef INTERACTION_H
#define INTERACTION_H

#include "ray.h"
#include "vec3.h"

class material;

struct hit_record {
    point3 p;
    vec3 normal;
    material *mat_ptr;
    double t;
    double u;
    double v;
    bool front_face;

    inline void set_face_normal(const ray &r, const vec3 &outWard_normal) {
        front_face = dot(r.direction(), outWard_normal) < 0;
        normal = front_face ? outWard_normal : -outWard_normal;
    }
};

struct BSDFSample {
    vec3 wi;
    color f;
    double pdf;
    bool is_specular;
    bool is_transmission = false;
};

#endif

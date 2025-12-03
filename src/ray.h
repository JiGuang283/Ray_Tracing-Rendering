#ifndef RAY_H
#define RAY_H

#include "vec3.h"

class ray {
  public:
    ray() = default;
    ray(const point3 &origin, const vec3 &direction, double time = 0.0) noexcept
        : orig(origin), dir(direction), tm(time) {
    }

    point3 origin() const noexcept {
        return orig;
    }
    vec3 direction() const noexcept {
        return dir;
    }
    double time() const noexcept {
        return tm;
    }

    point3 at(double t) const noexcept {
        return orig + t * dir;
    }

  private:
    point3 orig;
    vec3 dir;
    double tm = 0.0;
};

#endif
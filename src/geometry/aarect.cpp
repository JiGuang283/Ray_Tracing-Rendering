#include "aarect.h"

bool xy_rect::hit(const ray &r, double t_min, double t_max,
                  hit_record &rec) const {
    auto t = (k - r.origin().z()) / r.direction().z();
    if (t < t_min || t > t_max) {
        return false;
    }
    auto x = r.origin().x() + t * r.direction().x();
    auto y = r.origin().y() + t * r.direction().y();
    if (x < x0 || x > x1 || y < y0 || y > y1) {
        return false;
    }
    rec.u = (x - x0) / (x1 - x0);
    rec.v = (y - y0) / (y1 - y0);
    rec.t = t;

    auto outward_normal = vec3(0, 0, 1);
    rec.set_face_normal(r, outward_normal);
    rec.dpdu = vec3(x1 - x0, 0, 0);
    rec.dpdv = vec3(0, y1 - y0, 0);
    rec.mat_ptr = mp.get();
    rec.p = r.at(t);
    return true;
}

bool xy_rect::occluded(const ray &r, double t_min, double t_max,
                       RNG & /*rng*/) const {
    const double t = (k - r.origin().z()) / r.direction().z();
    if (t < t_min || t > t_max) {
        return false;
    }
    const double x = r.origin().x() + t * r.direction().x();
    const double y = r.origin().y() + t * r.direction().y();
    return x >= x0 && x <= x1 && y >= y0 && y <= y1;
}

bool xz_rect::hit(const ray &r, double t_min, double t_max,
                  hit_record &rec) const {
    auto t = (k - r.origin().y()) / r.direction().y();
    if (t < t_min || t > t_max) {
        return false;
    }
    auto x = r.origin().x() + t * r.direction().x();
    auto z = r.origin().z() + t * r.direction().z();
    if (x < x0 || x > x1 || z < z0 || z > z1) {
        return false;
    }
    rec.u = (x - x0) / (x1 - x0);
    rec.v = (z - z0) / (z1 - z0);
    rec.t = t;
    vec3 outward_normal = vec3(0, 1, 0);
    rec.set_face_normal(r, outward_normal);
    rec.dpdu = vec3(x1 - x0, 0, 0);
    rec.dpdv = vec3(0, 0, z1 - z0);
    rec.mat_ptr = mp.get();
    rec.p = r.at(t);
    return true;
}

bool xz_rect::occluded(const ray &r, double t_min, double t_max,
                       RNG & /*rng*/) const {
    const double t = (k - r.origin().y()) / r.direction().y();
    if (t < t_min || t > t_max) {
        return false;
    }
    const double x = r.origin().x() + t * r.direction().x();
    const double z = r.origin().z() + t * r.direction().z();
    return x >= x0 && x <= x1 && z >= z0 && z <= z1;
}

bool yz_rect::hit(const ray &r, double t_min, double t_max,
                  hit_record &rec) const {
    auto t = (k - r.origin().x()) / r.direction().x();
    if (t < t_min || t > t_max) {
        return false;
    }
    auto y = r.origin().y() + t * r.direction().y();
    auto z = r.origin().z() + t * r.direction().z();
    if (y < y0 || y > y1 || z < z0 || z > z1) {
        return false;
    }
    rec.u = (y - y0) / (y1 - y0);
    rec.v = (z - z0) / (z1 - z0);
    rec.t = t;
    vec3 outward_normal = vec3(1, 0, 0);
    rec.set_face_normal(r, outward_normal);
    rec.dpdu = vec3(0, y1 - y0, 0);
    rec.dpdv = vec3(0, 0, z1 - z0);
    rec.mat_ptr = mp.get();
    rec.p = r.at(t);
    return true;
}
bool yz_rect::occluded(const ray &r, double t_min, double t_max,
                       RNG & /*rng*/) const {
    const double t = (k - r.origin().x()) / r.direction().x();
    if (t < t_min || t > t_max) {
        return false;
    }
    const double y = r.origin().y() + t * r.direction().y();
    const double z = r.origin().z() + t * r.direction().z();
    return y >= y0 && y <= y1 && z >= z0 && z <= z1;
}

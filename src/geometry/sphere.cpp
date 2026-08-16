#include "sphere.h"

namespace {

void set_sphere_derivatives(double radius, double u, double v,
                            hit_record &rec) {
    double theta = v * pi;
    double alpha = u * 2.0 * pi - pi;
    double sin_theta = sin(theta);
    double cos_theta = cos(theta);
    double sin_alpha = sin(alpha);
    double cos_alpha = cos(alpha);

    rec.dpdu =
        2.0 * pi * radius *
        vec3(-sin_alpha * sin_theta, 0.0, -cos_alpha * sin_theta);
    rec.dpdv =
        pi * radius *
        vec3(cos_alpha * cos_theta, sin_theta, -sin_alpha * cos_theta);
}

} // namespace

bool sphere::hit(const ray &r, double t_min, double t_max,
                 hit_record &rec) const {
    vec3 oc = r.origin() - center;
    auto a = r.direction().length_squared();
    auto half_b = dot(oc, r.direction());
    auto c = oc.length_squared() - radius * radius;

    auto discriminant = half_b * half_b - a * c;
    if (discriminant < 0) {
        return false;
    }
    auto sqrtd = sqrt(discriminant);

    auto root = (-half_b - sqrtd) / a;
    if (root < t_min || root > t_max) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || root > t_max) {
            return false;
        }
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    vec3 outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(r, outward_normal);
    get_sphere_uv(outward_normal, rec.u, rec.v);
    set_sphere_derivatives(radius, rec.u, rec.v, rec);
    rec.mat_ptr = mat_ptr.get();

    return true;
}

bool sphere::occluded(const ray &r, double t_min, double t_max,
                       RNG & /*rng*/) const {
    const vec3 oc = r.origin() - center;
    const double a = r.direction().length_squared();
    const double half_b = dot(oc, r.direction());
    const double c = oc.length_squared() - radius * radius;
    const double discriminant = half_b * half_b - a * c;
    if (discriminant < 0.0) {
        return false;
    }
    const double sqrtd = sqrt(discriminant);
    const double first = (-half_b - sqrtd) / a;
    if (first >= t_min && first <= t_max) {
        return true;
    }
    const double second = (-half_b + sqrtd) / a;
    return second >= t_min && second <= t_max;
}

bool sphere::bounding_box(double /*time0*/, double /*time1*/,
                          aabb &output_box) const {
    output_box = aabb(center - vec3(radius, radius, radius),
                      center + vec3(radius, radius, radius));
    return true;
}

void sphere::get_sphere_uv(const point3 &p, double &u, double &v) {
    auto theta = acos(-p.y());
    auto phi = atan2(-p.z(), p.x()) + pi;

    u = phi / (2 * pi);
    v = theta / pi;
}

#ifndef INTERACTION_H
#define INTERACTION_H

#include "ray.h"
#include "vec3.h"

class material;

enum BSDFSampleFlags {
    BSDF_NONE = 0,
    BSDF_DIFFUSE = 1 << 0,
    BSDF_GLOSSY = 1 << 1,
    BSDF_DELTA = 1 << 2,
    BSDF_REFLECTION = 1 << 3,
    BSDF_TRANSMISSION = 1 << 4,
    BSDF_PHASE = 1 << 5
};

inline bool has_flag(int flags, BSDFSampleFlags flag) {
    return (flags & flag) != 0;
}

struct hit_record {
    point3 p;
    vec3 geometric_normal{0, 0, 1};
    vec3 normal;
    vec3 dpdu{1, 0, 0};
    vec3 dpdv{0, 1, 0};
    material *mat_ptr;
    double t;
    double u;
    double v;
    bool front_face;
    int primitive_id = -1;
    int material_id = -1;

    inline void set_face_normal(const ray &r, const vec3 &outWard_normal) {
        front_face = dot(r.direction(), outWard_normal) < 0;
        geometric_normal = front_face ? outWard_normal : -outWard_normal;
        normal = geometric_normal;
    }
};

struct ShadingFrame {
    vec3 tangent;
    vec3 bitangent;
    vec3 normal;

    ShadingFrame() : tangent(1, 0, 0), bitangent(0, 1, 0), normal(0, 0, 1) {
    }

    explicit ShadingFrame(const vec3 &n) {
        build_from_normal(n);
    }

    void build_from_normal(const vec3 &n) {
        normal = unit_vector(n);
        vec3 helper =
            (fabs(normal.x()) > 0.9) ? vec3(0, 1, 0) : vec3(1, 0, 0);
        bitangent = unit_vector(cross(normal, helper));
        tangent = cross(bitangent, normal);
    }

    void build_from_tangent_space(const vec3 &n, const vec3 &dpdu,
                                  const vec3 &dpdv) {
        normal = unit_vector(n);
        tangent = dpdu - dot(dpdu, normal) * normal;
        if (tangent.near_zero()) {
            tangent = cross(dpdv, normal);
        }
        if (tangent.near_zero()) {
            build_from_normal(normal);
            return;
        }
        tangent = unit_vector(tangent);
        bitangent = cross(normal, tangent);
        if (dot(bitangent, dpdv) < 0.0) {
            tangent = -tangent;
            bitangent = -bitangent;
        }
    }

    vec3 to_world(const vec3 &local) const {
        return local.x() * tangent + local.y() * bitangent +
               local.z() * normal;
    }

    vec3 to_local(const vec3 &world) const {
        return vec3(dot(world, tangent), dot(world, bitangent),
                    dot(world, normal));
    }
};

struct SurfaceInteraction {
    point3 p;
    vec3 geometry_normal;
    vec3 shading_normal;
    vec3 dpdu;
    vec3 dpdv;
    ShadingFrame frame;
    material *mat_ptr;
    double t;
    double u;
    double v;
    bool front_face;
    int primitive_id;
    int material_id;

    SurfaceInteraction()
        : p(0, 0, 0), geometry_normal(0, 0, 1), shading_normal(0, 0, 1),
          dpdu(1, 0, 0), dpdv(0, 1, 0), frame(vec3(0, 0, 1)),
          mat_ptr(nullptr), t(0), u(0), v(0), front_face(true),
          primitive_id(-1), material_id(-1) {
    }

    explicit SurfaceInteraction(const hit_record &rec)
        : p(rec.p), geometry_normal(rec.geometric_normal),
          shading_normal(rec.normal), dpdu(rec.dpdu), dpdv(rec.dpdv),
          frame(rec.normal), mat_ptr(rec.mat_ptr), t(rec.t), u(rec.u),
          v(rec.v), front_face(rec.front_face),
          primitive_id(rec.primitive_id), material_id(rec.material_id) {
        frame.build_from_tangent_space(shading_normal, dpdu, dpdv);
    }

    void set_shading_normal(const vec3 &normal) {
        shading_normal = unit_vector(normal);
        if (dot(shading_normal, geometry_normal) < 0) {
            shading_normal = -shading_normal;
        }
        frame.build_from_tangent_space(shading_normal, dpdu, dpdv);
    }

    point3 offset_origin(const vec3 &direction) const {
        constexpr double kRayOriginOffset = 1e-4;
        double side = dot(direction, geometry_normal) >= 0.0 ? 1.0 : -1.0;
        return p + side * kRayOriginOffset * geometry_normal;
    }

    ray spawn_ray(const vec3 &direction, double time = 0.0) const {
        return ray(offset_origin(direction), direction, time);
    }

    ray spawn_ray_to(const point3 &target, double time = 0.0) const {
        return spawn_ray(target - p, time);
    }
};

struct BSDFSample {
    vec3 wi;
    color f;
    double pdf;
    double eta;
    int flags;

    BSDFSample()
        : wi(0, 0, 0), f(0, 0, 0), pdf(0.0), eta(1.0),
          flags(BSDF_NONE) {
    }

    bool is_delta() const {
        return has_flag(flags, BSDF_DELTA);
    }

    bool is_transmission() const {
        return has_flag(flags, BSDF_TRANSMISSION);
    }

    bool is_phase() const {
        return has_flag(flags, BSDF_PHASE);
    }
};

#endif

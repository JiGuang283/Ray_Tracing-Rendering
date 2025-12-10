#ifndef MATERIAL_H
#define MATERIAL_H

#include "hittable.h"
#include "onb.h"
#include "ray.h"
#include "rtweekend.h"
#include "texture.h"
#include <algorithm>

struct hit_record;

class material {
  public:
    virtual ~material() = default;
    virtual color emitted(double u, double v, const point3 &p) const {
        return color(0, 0, 0);
    }

    virtual bool is_specular() const {
        return false;
    }

    virtual color eval(const ray &r_in, const hit_record &rec,
                       const ray &scattered) const {
        return color(0, 0, 0);
    }

    virtual double pdf(const ray &r_in, const hit_record &rec,
                       const ray &scattered) const {
        return 0.0;
    }

    virtual bool scatter(const ray &r_in, const hit_record &rec, color &albedo,
                         ray &scattered, double &pdf_val) const {
        return false;
    }

    // 保留旧接口
    virtual bool scatter(const ray &r_in, const hit_record &rec,
                         color &attenuation, ray &scattered) const {
        return false;
    }
};

class lambertian : public material {
  public:
    lambertian(const color &a) : albedo(make_shared<solid_color>(a)) {
    }
    lambertian(shared_ptr<texture> a) : albedo(a) {
    }

    virtual bool scatter(const ray &r_in, const hit_record &rec,
                         color &attenuation, ray &scattered) const override {
        auto scatter_direction = rec.normal + random_unit_vector();

        if (scatter_direction.near_zero()) {
            scatter_direction = rec.normal;
        }
        scattered = ray(rec.p, scatter_direction, r_in.time());
        attenuation = albedo->value(rec.u, rec.v, rec.p);
        return true;
    }

    virtual double pdf(const ray &r_in, const hit_record &rec,
                       const ray &scattered) const override {
        auto cosine = dot(rec.normal, unit_vector(scattered.direction()));
        return cosine < 0 ? 0 : cosine / pi;
    }

    virtual color eval(const ray &r_in, const hit_record &rec,
                       const ray &scattered) const override {
        return albedo->value(rec.u, rec.v, rec.p) / pi * fmax(0, dot(rec.normal, unit_vector(scattered.direction())));
    }

  public:
    shared_ptr<texture> albedo;
};

class metal : public material {
  public:
    metal(const color &a, double f) : albedo(a), fuzz(f < 1 ? f : 1) {
    }
    virtual bool scatter(const ray &r_in, const hit_record &rec,
                         color &attenuation, ray &scattered) const override {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered =
            ray(rec.p, reflected + fuzz * random_in_unit_sphere(), r_in.time());
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }

  public:
    color albedo;
    double fuzz;
};

class dielectric : public material {
  public:
    dielectric(double index_of_refraction) : ir(index_of_refraction) {
    }

    virtual bool scatter(const ray &r_in, const hit_record &rec,
                         color &attenuation, ray &scattered) const override {
        attenuation = color(1.0, 1.0, 1.0);
        double refraction_ratio = rec.front_face ? (1.0 / ir) : ir;

        vec3 unit_direction = unit_vector(r_in.direction());
        double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
        double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0;
        vec3 direction;

        if (cannot_refract ||
            reflectance(cos_theta, refraction_ratio) > random_double()) {
            direction = reflect(unit_direction, rec.normal);
        } else {
            direction = refract(unit_direction, rec.normal, refraction_ratio);
        }

        scattered = ray(rec.p, direction, r_in.time());
        return true;
    }

  public:
    double ir;

  private:
    static double reflectance(double cosine, double ref_idx) {
        auto r0 = (1 - ref_idx) / (1 + ref_idx);
        r0 = r0 * r0;
        return r0 + (1 - r0) * pow((1 - cosine), 5);
    }
};

class diffuse_light : public material {
  public:
    diffuse_light(shared_ptr<texture> a) : emit(a) {
    }
    diffuse_light(color c) : emit(make_shared<solid_color>(c)) {
    }

    virtual bool scatter(const ray &r_in, const hit_record &rec,
                         color &attenuation, ray &scattered) const override {
        return false;
    }

    virtual color emitted(double u, double v, const point3 &p) const override {
        return emit->value(u, v, p);
    }

  public:
    shared_ptr<texture> emit;
};

class PBRMaterial : public material {
  public:
    PBRMaterial(shared_ptr<texture> a, shared_ptr<texture> r,
                shared_ptr<texture> m, shared_ptr<texture> n = nullptr)
        : albedo(a), roughness(r), metallic(m), normal_map(n) {
    }

    virtual bool scatter(const ray &r_in, const hit_record &rec,
                         color &attenuation, ray &scattered) const override {
        vec3 N = rec.normal;
        if (normal_map) {
            onb uvw;
            uvw.build_from_w(N);
            vec3 local_n = normal_map->value_normal(rec.u, rec.v, rec.p);
            N = unit_vector(uvw.local(local_n));
        }

        double rough = roughness->value_scalar(rec.u, rec.v, rec.p);
        double metal = metallic->value_scalar(rec.u, rec.v, rec.p);
        color base_color = albedo->value(rec.u, rec.v, rec.p);

        vec3 unit_dir = unit_vector(r_in.direction());
        vec3 reflected = reflect(unit_dir, N);

        if (random_double() < metal) {
            vec3 scatter_dir = reflected + rough * random_in_unit_sphere();
            scattered = ray(rec.p, scatter_dir, r_in.time());
            attenuation = base_color;
            return (dot(scattered.direction(), N) > 0);
        } else {
            vec3 scatter_dir = N + random_unit_vector();
            if (scatter_dir.near_zero())
                scatter_dir = N;
            scattered = ray(rec.p, scatter_dir, r_in.time());
            attenuation = base_color;
            return true;
        }
    }

    double DistributionGGX(vec3 N, vec3 H, double roughness) const {
        double a = roughness * roughness;
        double a2 = a * a;
        double NdotH = std::max(dot(N, H), 0.0);
        double NdotH2 = NdotH * NdotH;

        double nom = a2;
        double denom = (NdotH2 * (a2 - 1.0) + 1.0);
        denom = pi * denom * denom;

        return nom / denom;
    }

    double GeometrySchlickGGX(double NdotV, double roughness) const {
        double r = (roughness + 1.0);
        double k = (r * r) / 8.0;

        double nom = NdotV;
        double denom = NdotV * (1.0 - k) + k;

        return nom / denom;
    }

    double GeometrySmith(vec3 N, vec3 V, vec3 L, double roughness) const {
        double NdotV = std::max(dot(N, V), 0.0);
        double NdotL = std::max(dot(N, L), 0.0);
        double ggx2 = GeometrySchlickGGX(NdotV, roughness);
        double ggx1 = GeometrySchlickGGX(NdotL, roughness);

        return ggx1 * ggx2;
    }

    vec3 fresnelSchlick(double cosTheta, vec3 F0) const {
        return F0 + (vec3(1.0, 1.0, 1.0) - F0) * pow(1.0 - cosTheta, 5.0);
    }

    virtual color eval(const ray &r_in, const hit_record &rec,
                       const ray &scattered) const override {
        vec3 N = rec.normal;
        if (normal_map) {
            onb uvw;
            uvw.build_from_w(N);
            vec3 local_n = normal_map->value_normal(rec.u, rec.v, rec.p);
            N = unit_vector(uvw.local(local_n));
        }

        vec3 V = unit_vector(-r_in.direction());
        vec3 L = unit_vector(scattered.direction());
        vec3 H = unit_vector(V + L);

        double rough = roughness->value_scalar(rec.u, rec.v, rec.p);
        double metal = metallic->value_scalar(rec.u, rec.v, rec.p);
        color base_color = albedo->value(rec.u, rec.v, rec.p);

        vec3 F0 = vec3(0.04, 0.04, 0.04);
        vec3 metal_vec(metal, metal, metal);
        F0 = (vec3(1.0, 1.0, 1.0) - metal_vec) * F0 + metal_vec * base_color;

        double NDF = DistributionGGX(N, H, rough);
        double G = GeometrySmith(N, V, L, rough);
        vec3 F = fresnelSchlick(std::max(dot(H, V), 0.0), F0);

        vec3 numerator = NDF * G * F;
        double denominator = 4.0 * std::max(dot(N, V), 0.0) *
                                 std::max(dot(N, L), 0.0) +
                             0.0001;
        vec3 specular = numerator / denominator;

        vec3 kS = F;
        vec3 kD = vec3(1.0, 1.0, 1.0) - kS;
        kD *= (1.0 - metal);

        double NdotL = std::max(dot(N, L), 0.0);

        return (kD * base_color / pi + specular) * NdotL;
    }

  public:
    shared_ptr<texture> albedo;
    shared_ptr<texture> roughness;
    shared_ptr<texture> metallic;
    shared_ptr<texture> normal_map;
};

#endif
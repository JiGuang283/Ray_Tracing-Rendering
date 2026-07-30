#include "material.h"

#include <algorithm>

namespace {

ShadingFrame make_normal_mapped_frame(const SurfaceInteraction &surface,
                                      const shared_ptr<texture> &normal_map) {
    if (!normal_map) {
        return surface.frame;
    }

    vec3 local_normal =
        normal_map->value_normal(surface.u, surface.v, surface.p);
    if (local_normal.near_zero()) {
        return surface.frame;
    }

    vec3 mapped_normal = unit_vector(surface.frame.to_world(local_normal));
    if (dot(mapped_normal, surface.geometry_normal) < 0.0) {
        mapped_normal = -mapped_normal;
    }
    return ShadingFrame(mapped_normal);
}

} // namespace

lambertian::lambertian(const color &a) : albedo(make_shared<solid_color>(a)) {
}

lambertian::lambertian(shared_ptr<texture> a) : albedo(a) {
}

void lambertian::shade(const SurfaceInteraction &surface,
                       ShadingResult &result) const {
    result.reset(surface.frame);
    result.bsdf.add_lambertian(albedo->value(surface.u, surface.v, surface.p));
}

metal::metal(const color &a, double f) : albedo(a), fuzz(f < 1 ? f : 1) {
}

void metal::shade(const SurfaceInteraction &surface,
                  ShadingResult &result) const {
    result.reset(surface.frame);
    if (fuzz <= 0.001) {
        result.bsdf.add_specular_reflection(albedo);
        return;
    }
    result.bsdf.add_microfacet_ggx(albedo, clamp(fuzz, 0.01, 1.0), 1.0);
}

dielectric::dielectric(double index_of_refraction)
    : ir(index_of_refraction) {
}

void dielectric::shade(const SurfaceInteraction &surface,
                       ShadingResult &result) const {
    result.reset(surface.frame);
    result.bsdf.add_specular_dielectric(ir, surface.front_face);
}

diffuse_light::diffuse_light(shared_ptr<texture> a) : emit(a) {
}

diffuse_light::diffuse_light(color c) : emit(make_shared<solid_color>(c)) {
}

void diffuse_light::shade(const SurfaceInteraction &surface,
                          ShadingResult &result) const {
    result.reset(surface.frame);
    if (surface.front_face) {
        result.set_emission(emit->value(surface.u, surface.v, surface.p));
    }
}

bool diffuse_light::is_emissive() const {
    return true;
}

color diffuse_light::emission_estimate() const {
    return emit->value(0.5, 0.5, point3(0, 0, 0));
}

PrincipledMaterial::PrincipledMaterial(shared_ptr<texture> base_color,
                                       shared_ptr<texture> roughness,
                                       shared_ptr<texture> metallic,
                                       shared_ptr<texture> normal_map,
                                       shared_ptr<texture> emission,
                                       double emission_strength,
                                       shared_ptr<texture> clearcoat,
                                       shared_ptr<texture> clearcoat_roughness)
    : base_color(base_color), roughness(roughness), metallic(metallic),
      normal_map(normal_map), emission(emission),
      emission_strength(emission_strength), clearcoat(clearcoat),
      clearcoat_roughness(clearcoat_roughness) {
}

void PrincipledMaterial::shade(const SurfaceInteraction &surface,
                               ShadingResult &result) const {
    ShadingFrame frame = make_normal_mapped_frame(surface, normal_map);
    result.reset(frame);

    color base = base_color->value(surface.u, surface.v, surface.p);
    double rough = roughness->value_roughness(surface.u, surface.v, surface.p);
    double metalness =
        metallic->value_metallic(surface.u, surface.v, surface.p);
    rough = clamp(rough, 0.01, 1.0);
    metalness = clamp(metalness, 0.0, 1.0);

    double diffuse_weight = 1.0 - metalness;
    if (diffuse_weight > 0.0) {
        result.bsdf.add_lambertian(diffuse_weight * base, diffuse_weight);
    }
    result.bsdf.add_microfacet_ggx(base, rough, metalness,
                                   0.5 + 0.5 * metalness);
    if (clearcoat) {
        double coat = clamp(clearcoat->value_scalar(surface.u, surface.v,
                                                    surface.p),
                            0.0, 1.0);
        if (coat > 0.0) {
            double coat_roughness =
                clearcoat_roughness
                    ? clearcoat_roughness->value_roughness(surface.u, surface.v,
                                                           surface.p)
                    : 0.1;
            result.bsdf.add_clearcoat_ggx(clamp(coat_roughness, 0.01, 1.0),
                                          coat);
        }
    }

    if (emission && surface.front_face) {
        result.set_emission(emission_strength *
                            emission->value(surface.u, surface.v, surface.p));
    }
}

bool PrincipledMaterial::is_emissive() const {
    return emission != nullptr && emission_strength > 0.0;
}

color PrincipledMaterial::emission_estimate() const {
    if (!emission) {
        return color(0, 0, 0);
    }
    return emission_strength * emission->value(0.5, 0.5, point3(0, 0, 0));
}

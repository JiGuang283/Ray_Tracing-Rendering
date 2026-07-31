#ifndef MATERIAL_PROGRAMS_H
#define MATERIAL_PROGRAMS_H

#include "material.h"
#include "normal_mapping.h"

MaterialHandle make_lambertian_material(TextureHandle albedo);
MaterialHandle make_lambertian_material(const color &albedo);
MaterialHandle make_metal_material(const color &albedo, double roughness);
MaterialHandle make_dielectric_material(double ior);
MaterialHandle make_diffuse_light_material(TextureHandle emission);
MaterialHandle make_diffuse_light_material(const color &emission);
MaterialHandle make_principled_material(
    TextureHandle base_color, TextureHandle roughness, TextureHandle metallic,
    TextureHandle normal_map = nullptr, TextureHandle emission = nullptr,
    double emission_strength = 1.0, TextureHandle clearcoat = nullptr,
    TextureHandle clearcoat_roughness = nullptr,
    NormalMapSettings normal_settings = {}, bool double_sided = false);
MaterialHandle make_isotropic_material(TextureHandle albedo);
MaterialHandle make_isotropic_material(const color &albedo);

#endif

#ifndef MATERIAL_H
#define MATERIAL_H

#include "shading/shading.h"
#include "shading/normal_mapping.h"
#include "texture.h"

class material {
  public:
    virtual ~material() = default;
    virtual void shade(const SurfaceInteraction &surface,
                       ShadingResult &result) const = 0;
    virtual bool is_emissive() const {
        return false;
    }
    virtual color emission_estimate() const {
        return color(0, 0, 0);
    }
};

class lambertian : public material {
  public:
    explicit lambertian(const color &a);
    explicit lambertian(TextureHandle a);

    void shade(const SurfaceInteraction &surface,
               ShadingResult &result) const override;

  private:
    TextureHandle albedo;
};

class metal : public material {
  public:
    metal(const color &a, double f);

    void shade(const SurfaceInteraction &surface,
               ShadingResult &result) const override;

  private:
    color albedo;
    double fuzz;
};

class dielectric : public material {
  public:
    explicit dielectric(double index_of_refraction);

    void shade(const SurfaceInteraction &surface,
               ShadingResult &result) const override;

  private:
    double ir;
};

class diffuse_light : public material {
  public:
    explicit diffuse_light(TextureHandle a);
    explicit diffuse_light(color c);

    void shade(const SurfaceInteraction &surface,
               ShadingResult &result) const override;
    bool is_emissive() const override;
    color emission_estimate() const override;

  private:
    TextureHandle emit;
};

class PrincipledMaterial : public material {
  public:
    PrincipledMaterial(TextureHandle base_color,
                       TextureHandle roughness,
                       TextureHandle metallic,
                       TextureHandle normal_map = nullptr,
                       TextureHandle emission = nullptr,
                       double emission_strength = 1.0,
                       TextureHandle clearcoat = nullptr,
                       TextureHandle clearcoat_roughness = nullptr,
                       NormalMapSettings normal_settings = {});

    void shade(const SurfaceInteraction &surface,
               ShadingResult &result) const override;
    bool is_emissive() const override;
    color emission_estimate() const override;

  private:
    TextureHandle base_color;
    TextureHandle roughness;
    TextureHandle metallic;
    TextureHandle normal_map;
    TextureHandle emission;
    double emission_strength;
    TextureHandle clearcoat;
    TextureHandle clearcoat_roughness;
    NormalMapSettings normal_settings;
};

using PBRMaterial = PrincipledMaterial;

#endif

#ifndef MATERIAL_H
#define MATERIAL_H

#include "shading/shading.h"
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
    explicit lambertian(shared_ptr<texture> a);

    void shade(const SurfaceInteraction &surface,
               ShadingResult &result) const override;

  private:
    shared_ptr<texture> albedo;
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
    explicit diffuse_light(shared_ptr<texture> a);
    explicit diffuse_light(color c);

    void shade(const SurfaceInteraction &surface,
               ShadingResult &result) const override;
    bool is_emissive() const override;
    color emission_estimate() const override;

  private:
    shared_ptr<texture> emit;
};

class PrincipledMaterial : public material {
  public:
    PrincipledMaterial(shared_ptr<texture> base_color,
                       shared_ptr<texture> roughness,
                       shared_ptr<texture> metallic,
                       shared_ptr<texture> normal_map = nullptr,
                       shared_ptr<texture> emission = nullptr,
                       double emission_strength = 1.0,
                       shared_ptr<texture> clearcoat = nullptr,
                       shared_ptr<texture> clearcoat_roughness = nullptr);

    void shade(const SurfaceInteraction &surface,
               ShadingResult &result) const override;
    bool is_emissive() const override;
    color emission_estimate() const override;

  private:
    shared_ptr<texture> base_color;
    shared_ptr<texture> roughness;
    shared_ptr<texture> metallic;
    shared_ptr<texture> normal_map;
    shared_ptr<texture> emission;
    double emission_strength;
    shared_ptr<texture> clearcoat;
    shared_ptr<texture> clearcoat_roughness;
};

using PBRMaterial = PrincipledMaterial;

#endif

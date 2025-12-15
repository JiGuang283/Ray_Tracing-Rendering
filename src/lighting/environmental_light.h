// lighting/environment_light.h
#ifndef ENVIRONMENT_LIGHT_H
#define ENVIRONMENT_LIGHT_H

#include "light.h"
#include "rtw_stb_image.h"

class EnvironmentLight : public Light {
  public:
    EnvironmentLight(const char *map_filename) {
        int components_per_pixel = 3;
        float *data =
            stbi_loadf(map_filename, &width, &height, &components_per_pixel, 0);

        if (!data) {
            std::cerr << "ERROR: Could not load HDR environment map: "
                      << map_filename << std::endl;
            width = height = 0;
        }

        if (data) {
            hdr_data = std::vector<float>(data, data + width * height * 3);
            stbi_image_free(data);
        }
    }

    virtual LightSample sample(const point3 &p, const vec2 &u) const override {
        LightSample s;
        s.wi = random_unit_vector();
        s.dist = infinity;
        s.is_delta = false;

        s.pdf = 1.0 / (4.0 * pi);

        s.Li = Le(ray(p, s.wi));

        return s;
    }

    virtual color Le(const ray &r) const override {
        if (hdr_data.empty())
            return color(1, 1, 1);

        vec3 unit_dir = unit_vector(r.direction());

        // u = phi / 2pi, v = theta / pi
        auto theta = acos(-unit_dir.y());
        auto phi = atan2(-unit_dir.z(), unit_dir.x()) + pi;

        double u = phi / (2 * pi);
        double v = theta / pi;

        int i = static_cast<int>(u * width);
        int j = static_cast<int>(v * height);

        // Clamp
        if (i >= width)
            i = width - 1;
        if (j >= height)
            j = height - 1;
        if (i < 0)
            i = 0;
        if (j < 0)
            j = 0;

        int index = 3 * (j * width + i);
        return color(hdr_data[index], hdr_data[index + 1], hdr_data[index + 2]);
    }

    // PDF
    virtual double pdf(const point3 &origin,
                       const vec3 &direction) const override {
        return 1.0 / (4.0 * pi);
    }

    virtual bool is_delta() const override {
        return false;
    }
    virtual bool is_infinite() const override {
        return true;
    }

  private:
    std::vector<float> hdr_data;
    int width, height;
};

#endif

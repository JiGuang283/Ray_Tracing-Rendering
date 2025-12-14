// lighting/environment_light.h
#ifndef ENVIRONMENT_LIGHT_H
#define ENVIRONMENT_LIGHT_H

#include "light.h"
#include "rtw_stb_image.h" // 确保路径对，或者用你自己的纹理加载类

class EnvironmentLight : public Light {
public:
    // map_filename: HDR 图片路径
    EnvironmentLight(const char* map_filename) {
        // 使用 rtw_stb_image 或者你现有的 image_texture 加载逻辑
        // 这里为了独立性，假设我们用 rtw_image 加载浮点数据
        int components_per_pixel = 3;
        float* data = stbi_loadf(map_filename, &width, &height, &components_per_pixel, 0);

        if (!data) {
            std::cerr << "ERROR: Could not load HDR environment map: " << map_filename << std::endl;
            width = height = 0;
        }

        // 把数据存起来 (简单起见存为 float 数组)
        // 实际工程中最好封装成 Texture 类
        if (data) {
            hdr_data = std::vector<float>(data, data + width * height * 3);
            stbi_image_free(data);
        }
    }

    // 1. 采样环境光
    // 这里的策略是：均匀采样整个球面 (Uniform Sphere Sampling)
    // 进阶策略是：重要性采样 (根据图片亮度构建 CDF)，那是加分项，先做基础的。
    virtual LightSample sample(const point3& p, const vec2& u) const override {
        LightSample s;
        // 随机生成一个方向 (均匀分布在球面上)
        s.wi = random_unit_vector();
        s.dist = infinity; // 环境光在无穷远处
        s.is_delta = false; // 环境光是面光源(无限大)，不是 Delta

        // PDF = 1 / (4 * pi)  (均匀分布的立体角 PDF)
        s.pdf = 1.0 / (4.0 * pi);

        // 获取该方向的颜色
        s.Li = Le(ray(p, s.wi));

        return s;
    }

    // 2. 背景光查询 (当光线没有打到物体时调用)
    virtual color Le(const ray& r) const override {
        if (hdr_data.empty()) return color(1, 1, 1); // 默认白色

        vec3 unit_dir = unit_vector(r.direction());

        // 球面坐标映射 UV
        // u = phi / 2pi, v = theta / pi
        auto theta = acos(-unit_dir.y());
        auto phi = atan2(-unit_dir.z(), unit_dir.x()) + pi;

        double u = phi / (2 * pi);
        double v = theta / pi;

        // 双线性插值读取颜色 (这里简化为最近邻)
        int i = static_cast<int>(u * width);
        int j = static_cast<int>(v * height);

        // Clamp
        if (i >= width) i = width - 1;
        if (j >= height) j = height - 1;
        if (i < 0) i = 0;
        if (j < 0) j = 0;

        int index = 3 * (j * width + i);
        return color(hdr_data[index], hdr_data[index + 1], hdr_data[index + 2]);
    }

    // PDF 
    virtual double pdf(const point3& origin, const vec3& direction) const override {
        return 1.0 / (4.0 * pi);
    }

    virtual bool is_delta() const override { return false; }

private:
    std::vector<float> hdr_data;
    int width, height;
};

#endif#pragma once

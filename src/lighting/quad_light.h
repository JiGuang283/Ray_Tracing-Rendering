#ifndef QUAD_LIGHT_H
#define QUAD_LIGHT_H

#include "light.h"


class QuadLight : public Light {
public:
    // corner: 矩形的一个角点
    // u, v: 矩形的两个边向量 (决定了大小和方向)
    // color: 发光颜色
    QuadLight(point3 corner, vec3 u, vec3 v, color c)
        : origin(corner), u(u), v(v), intensity(c) {
        area = cross(u, v).length();
    }

    virtual LightSample sample(const point3& p, const vec2& rnd) const override {
        LightSample s;

        // 1. 在矩形内部随机采样一个点
        // rnd 是传入的两个 [0,1) 随机数
        point3 light_point = origin + u * rnd.x() + v * rnd.y();

        // 2. 计算方向和距离
        vec3 d = light_point - p;
        double dist_sq = d.length_squared();
        s.dist = std::sqrt(dist_sq);
        s.wi = d / s.dist; // 指向光源采样点的方向

        s.is_delta = false; // 面光源不是 Delta 分布，它可以被随机打中

        // 3. 计算 PDF
        // 对于面光源，pdf = distance^2 / (area * cos_theta_light)
        // 这里我们要计算光线射出方向与矩形法线的夹角
        vec3 normal = unit_vector(cross(u, v));
        double cos_theta = dot(-s.wi, normal);

        if (cos_theta < 0.001) {
            // 如果是从背面看灯，或者光线平行于灯面，贡献极小
            s.pdf = 0;
            s.Li = color(0, 0, 0);
            return s;
        }

        // 立体角 PDF 转换公式
        s.pdf = dist_sq / (area * cos_theta);

        // 4. 辐射亮度
        s.Li = intensity;

        return s;
    }

    virtual bool is_delta() const override { return false; }

private:
    point3 origin;
    vec3 u, v;
    color intensity;
    double area;
};

#endif
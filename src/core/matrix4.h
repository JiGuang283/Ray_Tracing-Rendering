#ifndef MATRIX4_H
#define MATRIX4_H

#include "vec3.h"

#include <array>
#include <cstddef>

struct Quaternion {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double w = 1.0;

    Quaternion normalized() const;
};

class Matrix4 {
  public:
    Matrix4();
    explicit Matrix4(const std::array<double, 16> &values);

    static Matrix4 identity();
    static Matrix4 translation(const vec3 &offset);
    static Matrix4 scale(const vec3 &factors);
    static Matrix4 rotation(const Quaternion &rotation);
    static Matrix4 rotation_y(double angle_degrees);
    static Matrix4 from_column_major(const double *values);

    double operator()(std::size_t row, std::size_t column) const;
    double &operator()(std::size_t row, std::size_t column);

    Matrix4 transposed() const;
    Matrix4 inverse() const;
    double linear_determinant() const;

    point3 transform_point(const point3 &point) const;
    vec3 transform_vector(const vec3 &vector) const;

  private:
    std::array<double, 16> m_values;
};

Matrix4 operator*(const Matrix4 &left, const Matrix4 &right);

#endif

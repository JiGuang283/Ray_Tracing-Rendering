#include "matrix4.h"

#include "rtweekend.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

Quaternion Quaternion::normalized() const {
    const double length = std::sqrt(x * x + y * y + z * z + w * w);
    if (length <= 1e-15) {
        throw std::invalid_argument("Quaternion must be non-zero.");
    }
    return {x / length, y / length, z / length, w / length};
}

Matrix4::Matrix4()
    : m_values{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1} {
}

Matrix4::Matrix4(const std::array<double, 16> &values) : m_values(values) {
}

Matrix4 Matrix4::identity() {
    return Matrix4();
}

Matrix4 Matrix4::translation(const vec3 &offset) {
    Matrix4 result;
    result(0, 3) = offset.x();
    result(1, 3) = offset.y();
    result(2, 3) = offset.z();
    return result;
}

Matrix4 Matrix4::scale(const vec3 &factors) {
    Matrix4 result;
    result(0, 0) = factors.x();
    result(1, 1) = factors.y();
    result(2, 2) = factors.z();
    return result;
}

Matrix4 Matrix4::rotation(const Quaternion &rotation) {
    const Quaternion q = rotation.normalized();
    const double xx = q.x * q.x;
    const double yy = q.y * q.y;
    const double zz = q.z * q.z;
    const double xy = q.x * q.y;
    const double xz = q.x * q.z;
    const double yz = q.y * q.z;
    const double xw = q.x * q.w;
    const double yw = q.y * q.w;
    const double zw = q.z * q.w;

    return Matrix4({1.0 - 2.0 * (yy + zz), 2.0 * (xy - zw),
                    2.0 * (xz + yw), 0.0, 2.0 * (xy + zw),
                    1.0 - 2.0 * (xx + zz), 2.0 * (yz - xw), 0.0,
                    2.0 * (xz - yw), 2.0 * (yz + xw),
                    1.0 - 2.0 * (xx + yy), 0.0, 0.0, 0.0, 0.0, 1.0});
}

Matrix4 Matrix4::rotation_y(double angle_degrees) {
    const double angle = degrees_to_radians(angle_degrees);
    const double half_angle = 0.5 * angle;
    return rotation({0.0, std::sin(half_angle), 0.0,
                     std::cos(half_angle)});
}

Matrix4 Matrix4::from_column_major(const double *values) {
    Matrix4 result;
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t column = 0; column < 4; ++column) {
            result(row, column) = values[column * 4 + row];
        }
    }
    return result;
}

double Matrix4::operator()(std::size_t row, std::size_t column) const {
    return m_values[row * 4 + column];
}

double &Matrix4::operator()(std::size_t row, std::size_t column) {
    return m_values[row * 4 + column];
}

Matrix4 Matrix4::transposed() const {
    Matrix4 result;
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t column = 0; column < 4; ++column) {
            result(row, column) = (*this)(column, row);
        }
    }
    return result;
}

Matrix4 Matrix4::inverse() const {
    double augmented[4][8]{};
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t column = 0; column < 4; ++column) {
            augmented[row][column] = (*this)(row, column);
        }
        augmented[row][row + 4] = 1.0;
    }

    for (std::size_t column = 0; column < 4; ++column) {
        std::size_t pivot = column;
        for (std::size_t row = column + 1; row < 4; ++row) {
            if (std::abs(augmented[row][column]) >
                std::abs(augmented[pivot][column])) {
                pivot = row;
            }
        }
        if (std::abs(augmented[pivot][column]) <= 1e-15) {
            throw std::invalid_argument("Transform matrix is singular.");
        }
        if (pivot != column) {
            for (std::size_t entry = 0; entry < 8; ++entry) {
                std::swap(augmented[pivot][entry],
                          augmented[column][entry]);
            }
        }

        const double divisor = augmented[column][column];
        for (double &entry : augmented[column]) {
            entry /= divisor;
        }
        for (std::size_t row = 0; row < 4; ++row) {
            if (row == column) {
                continue;
            }
            const double factor = augmented[row][column];
            for (std::size_t entry = 0; entry < 8; ++entry) {
                augmented[row][entry] -= factor * augmented[column][entry];
            }
        }
    }

    Matrix4 result;
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t column = 0; column < 4; ++column) {
            result(row, column) = augmented[row][column + 4];
        }
    }
    return result;
}

double Matrix4::linear_determinant() const {
    return (*this)(0, 0) *
               ((*this)(1, 1) * (*this)(2, 2) -
                (*this)(1, 2) * (*this)(2, 1)) -
           (*this)(0, 1) *
               ((*this)(1, 0) * (*this)(2, 2) -
                (*this)(1, 2) * (*this)(2, 0)) +
           (*this)(0, 2) *
               ((*this)(1, 0) * (*this)(2, 1) -
                (*this)(1, 1) * (*this)(2, 0));
}

point3 Matrix4::transform_point(const point3 &point) const {
    const double x = (*this)(0, 0) * point.x() +
                     (*this)(0, 1) * point.y() +
                     (*this)(0, 2) * point.z() + (*this)(0, 3);
    const double y = (*this)(1, 0) * point.x() +
                     (*this)(1, 1) * point.y() +
                     (*this)(1, 2) * point.z() + (*this)(1, 3);
    const double z = (*this)(2, 0) * point.x() +
                     (*this)(2, 1) * point.y() +
                     (*this)(2, 2) * point.z() + (*this)(2, 3);
    const double w = (*this)(3, 0) * point.x() +
                     (*this)(3, 1) * point.y() +
                     (*this)(3, 2) * point.z() + (*this)(3, 3);
    if (std::abs(w) <= 1e-15) {
        throw std::runtime_error("Point transformed to zero homogeneous w.");
    }
    return point3(x / w, y / w, z / w);
}

vec3 Matrix4::transform_vector(const vec3 &vector) const {
    return vec3((*this)(0, 0) * vector.x() +
                    (*this)(0, 1) * vector.y() +
                    (*this)(0, 2) * vector.z(),
                (*this)(1, 0) * vector.x() +
                    (*this)(1, 1) * vector.y() +
                    (*this)(1, 2) * vector.z(),
                (*this)(2, 0) * vector.x() +
                    (*this)(2, 1) * vector.y() +
                    (*this)(2, 2) * vector.z());
}

Matrix4 operator*(const Matrix4 &left, const Matrix4 &right) {
    Matrix4 result;
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t column = 0; column < 4; ++column) {
            result(row, column) = 0.0;
            for (std::size_t inner = 0; inner < 4; ++inner) {
                result(row, column) +=
                    left(row, inner) * right(inner, column);
            }
        }
    }
    return result;
}

#ifndef TRIANGLE_INTERSECTION_H
#define TRIANGLE_INTERSECTION_H

#include "host_device.h"

#include <cmath>
#include <limits>

namespace triangle_intersection {

template <typename T> struct TriangleKernelVector {
    T x = T(0);
    T y = T(0);
    T z = T(0);
};

template <typename T> struct TriangleKernelHit {
    T t = T(0);
    T barycentric_u = T(0);
    T barycentric_v = T(0);
};

template <typename T>
RT_HOST_DEVICE RT_FORCE_INLINE TriangleKernelVector<T>
subtract(TriangleKernelVector<T> left, TriangleKernelVector<T> right) {
    return {left.x - right.x, left.y - right.y, left.z - right.z};
}

template <typename T>
RT_HOST_DEVICE RT_FORCE_INLINE T dot(TriangleKernelVector<T> left,
                                     TriangleKernelVector<T> right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

template <typename T>
RT_HOST_DEVICE RT_FORCE_INLINE TriangleKernelVector<T>
cross(TriangleKernelVector<T> left, TriangleKernelVector<T> right) {
    return {left.y * right.z - left.z * right.y,
            left.z * right.x - left.x * right.z,
            left.x * right.y - left.y * right.x};
}

template <typename T>
RT_HOST_DEVICE RT_FORCE_INLINE T absolute(T value) {
    return value < T(0) ? -value : value;
}

template <typename T>
RT_HOST_DEVICE RT_FORCE_INLINE bool finite(T value) {
    const T maximum = std::numeric_limits<T>::max();
    return value == value && value <= maximum && value >= -maximum;
}

template <typename T>
RT_HOST_DEVICE RT_FORCE_INLINE bool finite(TriangleKernelVector<T> value) {
    return finite(value.x) && finite(value.y) && finite(value.z);
}

template <typename T>
RT_HOST_DEVICE RT_FORCE_INLINE T square_root(T value) {
    if constexpr (sizeof(T) == sizeof(float)) {
        return static_cast<T>(::sqrtf(static_cast<float>(value)));
    }
    return static_cast<T>(::sqrt(static_cast<double>(value)));
}

template <typename T>
RT_HOST_DEVICE RT_FORCE_INLINE bool intersect_triangle_kernel(
    TriangleKernelVector<T> origin, TriangleKernelVector<T> direction,
    TriangleKernelVector<T> vertex0, TriangleKernelVector<T> vertex1,
    TriangleKernelVector<T> vertex2, T t_min, T t_max,
    TriangleKernelHit<T> &hit) {
    constexpr T parallel_tolerance = static_cast<T>(1e-7);
    if (!finite(origin) || !finite(direction) || !finite(vertex0) ||
        !finite(vertex1) || !finite(vertex2) || t_min != t_min ||
        t_max != t_max || t_max < t_min) {
        return false;
    }

    const TriangleKernelVector<T> edge1 = subtract(vertex1, vertex0);
    const TriangleKernelVector<T> edge2 = subtract(vertex2, vertex0);
    const TriangleKernelVector<T> triangle_normal = cross(edge1, edge2);
    const T normal_length_squared = dot(triangle_normal, triangle_normal);
    const T direction_length_squared = dot(direction, direction);
    if (!(normal_length_squared > T(0)) ||
        !(direction_length_squared > T(0)) ||
        !finite(normal_length_squared) ||
        !finite(direction_length_squared)) {
        return false;
    }

    const TriangleKernelVector<T> pvec = cross(direction, edge2);
    const T determinant = dot(edge1, pvec);
    const T determinant_limit =
        parallel_tolerance * square_root(normal_length_squared) *
        square_root(direction_length_squared);
    if (!finite(determinant) || !finite(determinant_limit) ||
        absolute(determinant) <= determinant_limit) {
        return false;
    }

    const T inverse_determinant = T(1) / determinant;
    const TriangleKernelVector<T> tvec = subtract(origin, vertex0);
    const T barycentric_u = dot(tvec, pvec) * inverse_determinant;
    if (!finite(barycentric_u) || barycentric_u < T(0) ||
        barycentric_u > T(1)) {
        return false;
    }

    const TriangleKernelVector<T> qvec = cross(tvec, edge1);
    const T barycentric_v = dot(direction, qvec) * inverse_determinant;
    if (!finite(barycentric_v) || barycentric_v < T(0) ||
        barycentric_u + barycentric_v > T(1)) {
        return false;
    }

    const T t = dot(edge2, qvec) * inverse_determinant;
    if (!finite(t) || t < t_min || t > t_max) {
        return false;
    }

    hit.t = t;
    hit.barycentric_u = barycentric_u;
    hit.barycentric_v = barycentric_v;
    return true;
}

} // namespace triangle_intersection

#endif

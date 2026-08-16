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
RT_HOST_DEVICE RT_FORCE_INLINE T component(TriangleKernelVector<T> value,
                                           int axis) {
    return axis == 0 ? value.x : (axis == 1 ? value.y : value.z);
}

template <typename T>
RT_HOST_DEVICE RT_FORCE_INLINE T fused_multiply_add(T left, T right,
                                                    T addend) {
    if constexpr (sizeof(T) == sizeof(float)) {
        return static_cast<T>(::fmaf(static_cast<float>(left),
                                     static_cast<float>(right),
                                     static_cast<float>(addend)));
    }
    return left * right + addend;
}

template <typename T>
RT_HOST_DEVICE RT_FORCE_INLINE T difference_of_products(T left0, T left1,
                                                        T right0, T right1) {
    const T right_product = right0 * right1;
    const T right_error =
        fused_multiply_add(-right0, right1, right_product);
    const T difference =
        fused_multiply_add(left0, left1, -right_product);
    return difference + right_error;
}

template <typename T>
RT_HOST_DEVICE RT_FORCE_INLINE bool edge_near_zero(
    T value, T left0, T left1, T right0, T right1) {
    const T product_scale =
        absolute(left0 * left1) + absolute(right0 * right1);
    const T error_bound = T(8) * std::numeric_limits<T>::epsilon() *
                          product_scale;
    return absolute(value) <= error_bound;
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

    const T determinant = dot(triangle_normal, direction);
    const T determinant_limit =
        parallel_tolerance * square_root(normal_length_squared) *
        square_root(direction_length_squared);
    if (!finite(determinant) || !finite(determinant_limit) ||
        absolute(determinant) <= determinant_limit) {
        return false;
    }

    const TriangleKernelVector<T> translated0 = subtract(vertex0, origin);
    const TriangleKernelVector<T> translated1 = subtract(vertex1, origin);
    const TriangleKernelVector<T> translated2 = subtract(vertex2, origin);

    int dominant_axis = 0;
    if (absolute(direction.y) > absolute(direction.x)) {
        dominant_axis = 1;
    }
    if (absolute(direction.z) >
        absolute(component(direction, dominant_axis))) {
        dominant_axis = 2;
    }
    int first_axis = (dominant_axis + 1) % 3;
    int second_axis = (first_axis + 1) % 3;
    if (component(direction, dominant_axis) < T(0)) {
        const int temporary = first_axis;
        first_axis = second_axis;
        second_axis = temporary;
    }

    const T inverse_dominant = T(1) / component(direction, dominant_axis);
    const T shear_x = -component(direction, first_axis) * inverse_dominant;
    const T shear_y = -component(direction, second_axis) * inverse_dominant;
    const T shear_z = inverse_dominant;

    const T x0 = fused_multiply_add(
        shear_x, component(translated0, dominant_axis),
        component(translated0, first_axis));
    const T y0 = fused_multiply_add(
        shear_y, component(translated0, dominant_axis),
        component(translated0, second_axis));
    const T x1 = fused_multiply_add(
        shear_x, component(translated1, dominant_axis),
        component(translated1, first_axis));
    const T y1 = fused_multiply_add(
        shear_y, component(translated1, dominant_axis),
        component(translated1, second_axis));
    const T x2 = fused_multiply_add(
        shear_x, component(translated2, dominant_axis),
        component(translated2, first_axis));
    const T y2 = fused_multiply_add(
        shear_y, component(translated2, dominant_axis),
        component(translated2, second_axis));

    T edge0;
    T edge1_value;
    T edge2_value;
    if constexpr (sizeof(T) == sizeof(float)) {
        edge0 = difference_of_products(x1, y2, y1, x2);
        edge1_value = difference_of_products(x2, y0, y2, x0);
        edge2_value = difference_of_products(x0, y1, y0, x1);
        if (edge_near_zero(edge0, x1, y2, y1, x2) ||
            edge_near_zero(edge1_value, x2, y0, y2, x0) ||
            edge_near_zero(edge2_value, x0, y1, y0, x1)) {
            edge0 = static_cast<T>(static_cast<double>(x1) *
                                       static_cast<double>(y2) -
                                   static_cast<double>(y1) *
                                       static_cast<double>(x2));
            edge1_value =
                static_cast<T>(static_cast<double>(x2) *
                                   static_cast<double>(y0) -
                               static_cast<double>(y2) *
                                   static_cast<double>(x0));
            edge2_value =
                static_cast<T>(static_cast<double>(x0) *
                                   static_cast<double>(y1) -
                                   static_cast<double>(y0) *
                                       static_cast<double>(x1));
        }
    } else {
        edge0 = x1 * y2 - y1 * x2;
        edge1_value = x2 * y0 - y2 * x0;
        edge2_value = x0 * y1 - y0 * x1;
    }

    const bool has_negative =
        edge0 < T(0) || edge1_value < T(0) || edge2_value < T(0);
    const bool has_positive =
        edge0 > T(0) || edge1_value > T(0) || edge2_value > T(0);
    if (has_negative && has_positive) {
        return false;
    }

    const T projected_determinant = edge0 + edge1_value + edge2_value;
    if (projected_determinant == T(0) ||
        !finite(projected_determinant)) {
        return false;
    }

    const T z0 = shear_z * component(translated0, dominant_axis);
    const T z1 = shear_z * component(translated1, dominant_axis);
    const T z2 = shear_z * component(translated2, dominant_axis);
    const T scaled_t =
        edge0 * z0 + edge1_value * z1 + edge2_value * z2;
    if (!finite(scaled_t)) {
        return false;
    }
    if (projected_determinant > T(0)) {
        if (scaled_t < t_min * projected_determinant ||
            scaled_t > t_max * projected_determinant) {
            return false;
        }
    } else if (scaled_t > t_min * projected_determinant ||
               scaled_t < t_max * projected_determinant) {
        return false;
    }

    const T inverse_projected_determinant = T(1) / projected_determinant;
    const T t = scaled_t * inverse_projected_determinant;
    if (!finite(t) || t < t_min || t > t_max) {
        return false;
    }

    const T barycentric_u =
        edge1_value * inverse_projected_determinant;
    const T barycentric_v =
        edge2_value * inverse_projected_determinant;
    if (!finite(barycentric_u) || !finite(barycentric_v)) {
        return false;
    }

    hit.t = t;
    hit.barycentric_u = barycentric_u;
    hit.barycentric_v = barycentric_v;
    return true;
}

// Simplified Möller-Trumbore intersection used only by the CPU packed fast
// transport. It is intentionally less robust than intersect_triangle_kernel:
// callers must choose it only for host validation paths, never for CUDA.
template <typename T>
RT_HOST_DEVICE RT_FORCE_INLINE bool intersect_triangle_kernel_host_fast(
    TriangleKernelVector<T> origin, TriangleKernelVector<T> direction,
    TriangleKernelVector<T> vertex0, TriangleKernelVector<T> vertex1,
    TriangleKernelVector<T> vertex2, T t_min, T t_max,
    TriangleKernelHit<T> &hit) {
    const TriangleKernelVector<T> edge1 = subtract(vertex1, vertex0);
    const TriangleKernelVector<T> edge2 = subtract(vertex2, vertex0);
    const TriangleKernelVector<T> pvec = cross(direction, edge2);
    const T determinant = dot(edge1, pvec);
    const T epsilon = static_cast<T>(1e-7);
    if (!finite(determinant) || absolute(determinant) <= epsilon) {
        return false;
    }
    const T inverse = T(1) / determinant;
    const TriangleKernelVector<T> tvec = subtract(origin, vertex0);
    const T u = dot(tvec, pvec) * inverse;
    if (!finite(u) || u < T(0) || u > T(1)) {
        return false;
    }
    const TriangleKernelVector<T> qvec = cross(tvec, edge1);
    const T v = dot(direction, qvec) * inverse;
    if (!finite(v) || v < T(0) || u + v > T(1)) {
        return false;
    }
    const T t = dot(edge2, qvec) * inverse;
    if (!finite(t) || t < t_min || t > t_max) {
        return false;
    }
    hit.t = t;
    hit.barycentric_u = u;
    hit.barycentric_v = v;
    return true;
}

} // namespace triangle_intersection

#endif

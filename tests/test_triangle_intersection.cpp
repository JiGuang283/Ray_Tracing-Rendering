#include "test_harness.h"

#include "triangle_intersection.h"

#include <array>
#include <limits>

namespace {

template <typename T>
triangle_intersection::TriangleKernelHit<T> hit_scaled_triangle(T scale) {
    using namespace triangle_intersection;
    TriangleKernelHit<T> hit;
    REQUIRE(intersect_triangle_kernel(
        TriangleKernelVector<T>{T(0.25) * scale, T(0.25) * scale, scale},
        TriangleKernelVector<T>{T(0), T(0), -scale},
        TriangleKernelVector<T>{T(0), T(0), T(0)},
        TriangleKernelVector<T>{scale, T(0), T(0)},
        TriangleKernelVector<T>{T(0), scale, T(0)}, T(0), T(2), hit));
    return hit;
}

template <typename T>
void require_shared_edge_is_covered(T scale, T edge_parameter) {
    using namespace triangle_intersection;
    const TriangleKernelVector<T> origin{
        edge_parameter * scale, (T(1) - edge_parameter) * scale, scale};
    const TriangleKernelVector<T> direction{T(0), T(0), -scale};
    TriangleKernelHit<T> first;
    TriangleKernelHit<T> second;
    const bool first_found = intersect_triangle_kernel(
        origin, direction, {T(0), T(0), T(0)}, {scale, T(0), T(0)},
        {T(0), scale, T(0)}, T(0), T(2), first);
    const bool second_found = intersect_triangle_kernel(
        origin, direction, {scale, scale, T(0)}, {T(0), scale, T(0)},
        {scale, T(0), T(0)}, T(0), T(2), second);
    REQUIRE(first_found || second_found);
    if (first_found) {
        REQUIRE_NEAR(first.t, T(1), T(2e-5));
    }
    if (second_found) {
        REQUIRE_NEAR(second.t, T(1), T(2e-5));
    }
}

} // namespace

TEST_CASE(triangle_intersection_is_invariant_to_local_scale) {
    for (double scale : std::array<double, 3>{1e-4, 1.0, 1e4}) {
        const auto hit = hit_scaled_triangle(scale);
        REQUIRE_NEAR(hit.t, 1.0, 1e-12);
        REQUIRE_NEAR(hit.barycentric_u, 0.25, 1e-12);
        REQUIRE_NEAR(hit.barycentric_v, 0.25, 1e-12);
    }
    for (float scale : std::array<float, 3>{1e-4f, 1.0f, 1e4f}) {
        const auto hit = hit_scaled_triangle(scale);
        REQUIRE_NEAR(hit.t, 1.0, 2e-5);
        REQUIRE_NEAR(hit.barycentric_u, 0.25, 2e-5);
        REQUIRE_NEAR(hit.barycentric_v, 0.25, 2e-5);
    }
}

TEST_CASE(triangle_intersection_preserves_ray_parameterization) {
    using namespace triangle_intersection;
    for (double multiplier : std::array<double, 3>{1e-3, 1.0, 1e3}) {
        TriangleKernelHit<double> hit;
        REQUIRE(intersect_triangle_kernel(
            TriangleKernelVector<double>{0.2, 0.3, 1.0},
            TriangleKernelVector<double>{0.0, 0.0, -multiplier},
            TriangleKernelVector<double>{0.0, 0.0, 0.0},
            TriangleKernelVector<double>{1.0, 0.0, 0.0},
            TriangleKernelVector<double>{0.0, 1.0, 0.0}, 0.0, 2000.0,
            hit));
        REQUIRE_NEAR(hit.t, 1.0 / multiplier, 1e-9);
        REQUIRE_NEAR(hit.t * multiplier, 1.0, 1e-12);
        REQUIRE_NEAR(hit.barycentric_u, 0.2, 1e-12);
        REQUIRE_NEAR(hit.barycentric_v, 0.3, 1e-12);
    }
}

TEST_CASE(triangle_intersection_handles_winding_and_shared_edges) {
    using namespace triangle_intersection;
    const TriangleKernelVector<double> origin{0.5, 0.5, 1.0};
    const TriangleKernelVector<double> direction{0.0, 0.0, -1.0};
    TriangleKernelHit<double> first;
    TriangleKernelHit<double> second;
    REQUIRE(intersect_triangle_kernel(
        origin, direction, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0}, 0.0, 2.0, first));
    REQUIRE(intersect_triangle_kernel(
        origin, direction, {1.0, 1.0, 0.0}, {0.0, 1.0, 0.0},
        {1.0, 0.0, 0.0}, 0.0, 2.0, second));

    TriangleKernelHit<double> reversed;
    REQUIRE(intersect_triangle_kernel(
        {0.2, 0.3, 1.0}, direction, {0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0}, {1.0, 0.0, 0.0}, 0.0, 2.0, reversed));
    REQUIRE_NEAR(reversed.barycentric_u, 0.3, 1e-12);
    REQUIRE_NEAR(reversed.barycentric_v, 0.2, 1e-12);

    for (double scale : std::array<double, 3>{1e-4, 1.0, 1e4}) {
        for (double edge_parameter :
             std::array<double, 5>{0.0, 0.25, 0.5, 0.75, 1.0}) {
            require_shared_edge_is_covered(scale, edge_parameter);
        }
    }
    for (float scale : std::array<float, 3>{1e-4f, 1.0f, 1e4f}) {
        for (float edge_parameter :
             std::array<float, 5>{0.0f, 0.25f, 0.5f, 0.75f, 1.0f}) {
            require_shared_edge_is_covered(scale, edge_parameter);
        }
    }
}

TEST_CASE(triangle_intersection_rejects_invalid_and_out_of_range_rays) {
    using namespace triangle_intersection;
    const TriangleKernelVector<double> v0{0.0, 0.0, 0.0};
    const TriangleKernelVector<double> v1{1.0, 0.0, 0.0};
    const TriangleKernelVector<double> v2{0.0, 1.0, 0.0};
    TriangleKernelHit<double> hit;
    REQUIRE(!intersect_triangle_kernel(
        {0.25, 0.25, 1.0}, {1.0, 0.0, 0.0}, v0, v1, v2, 0.0, 2.0,
        hit));
    REQUIRE(!intersect_triangle_kernel(
        {0.25, 0.25, 1.0}, {0.0, 0.0, 0.0}, v0, v1, v2, 0.0, 2.0,
        hit));
    REQUIRE(!intersect_triangle_kernel(
        {0.25, 0.25, 1.0}, {0.0, 0.0, -1.0}, v0, v1, v1, 0.0, 2.0,
        hit));
    REQUIRE(!intersect_triangle_kernel(
        {0.25, 0.25, 1.0}, {0.0, 0.0, -1.0}, v0, v1, v2, 0.0, 0.5,
        hit));
    REQUIRE(intersect_triangle_kernel(
        {0.25, 0.25, 1.0}, {0.0, 0.0, -1.0}, v0, v1, v2, 1.0, 1.0,
        hit));
    REQUIRE(!intersect_triangle_kernel(
        {std::numeric_limits<double>::infinity(), 0.25, 1.0},
        {0.0, 0.0, -1.0}, v0, v1, v2, 0.0, 2.0, hit));
}

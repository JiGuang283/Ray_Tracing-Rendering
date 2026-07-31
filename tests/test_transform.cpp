#include "test_harness.h"

#include "transform.h"

namespace {

void require_vec_near(const vec3 &actual, const vec3 &expected,
                      double tolerance = 1e-9) {
    REQUIRE_NEAR(actual.x(), expected.x(), tolerance);
    REQUIRE_NEAR(actual.y(), expected.y(), tolerance);
    REQUIRE_NEAR(actual.z(), expected.z(), tolerance);
}

} // namespace

TEST_CASE(transform_trs_composition_and_inverse) {
    const Transform transform = Transform::from_trs(
        vec3(4, -2, 1), Quaternion{0, 0, 0, 1}, vec3(2, 3, 4));
    const point3 world = transform.point_to_world(point3(1, 2, 3));
    require_vec_near(world, point3(6, 4, 13));
    require_vec_near(transform.point_to_object(world), point3(1, 2, 3));
}

TEST_CASE(transform_preserves_ray_parameter) {
    const Transform transform =
        Transform::translate(vec3(3, 1, -2)) *
        Transform::scale(vec3(2, 3, 4));
    const ray world(point3(7, 7, 10), vec3(2, -3, 4), 0.25);
    const ray object = transform.ray_to_object(world);
    require_vec_near(transform.point_to_world(object.at(2.5)), world.at(2.5));
    REQUIRE_NEAR(object.time(), 0.25, 1e-12);
}

TEST_CASE(transform_uses_inverse_transpose_for_normals) {
    const Transform transform = Transform::scale(vec3(2, 1, 0.5));
    const vec3 tangent = transform.vector_to_world(vec3(1, 0, 1));
    const vec3 normal = transform.normal_to_world(vec3(1, 0, -1));
    REQUIRE_NEAR(dot(tangent, normal), 0.0, 1e-12);
}

TEST_CASE(transform_tracks_mirrored_handedness) {
    REQUIRE(!Transform::scale(vec3(2, 3, 4)).swaps_handedness());
    REQUIRE(Transform::scale(vec3(-2, 3, 4)).swaps_handedness());
}

TEST_CASE(transform_expands_world_bounds) {
    const Transform transform = Transform::translate(vec3(2, 0, 0)) *
                                Transform::rotate_y(90.0);
    const aabb world =
        transform.bounds_to_world(aabb(point3(0, 0, 0), point3(1, 2, 3)));
    require_vec_near(world.min(), point3(2, 0, -1), 1e-9);
    require_vec_near(world.max(), point3(5, 2, 0), 1e-9);
}

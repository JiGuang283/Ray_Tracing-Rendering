#include "sample_filter.h"
#include "test_harness.h"

#include <limits>

namespace {

double luminance(const color &value) {
    return 0.2126 * value.x() + 0.7152 * value.y() +
           0.0722 * value.z();
}

} // namespace

TEST_CASE(camera_sample_clamp_is_disabled_by_zero) {
    const color radiance(1000, 200, 50);
    const FilteredCameraSample result = filter_camera_sample(radiance, 0.0);
    REQUIRE(!result.clamped);
    REQUIRE(!result.invalid);
    REQUIRE_NEAR(result.radiance.x(), radiance.x(), 1e-12);
    REQUIRE_NEAR(result.radiance.y(), radiance.y(), 1e-12);
    REQUIRE_NEAR(result.radiance.z(), radiance.z(), 1e-12);
}

TEST_CASE(camera_sample_clamp_limits_luminance_and_preserves_color_ratio) {
    const color radiance(20, 10, 5);
    const FilteredCameraSample result = filter_camera_sample(radiance, 2.0);
    REQUIRE(result.clamped);
    REQUIRE(!result.invalid);
    REQUIRE_NEAR(luminance(result.radiance), 2.0, 1e-12);
    REQUIRE_NEAR(result.radiance.x() / result.radiance.y(), 2.0, 1e-12);
    REQUIRE_NEAR(result.radiance.y() / result.radiance.z(), 2.0, 1e-12);
}

TEST_CASE(non_finite_camera_samples_are_discarded) {
    const double infinity_value = std::numeric_limits<double>::infinity();
    const FilteredCameraSample result =
        filter_camera_sample(color(1, infinity_value, 3), 0.0);
    REQUIRE(result.invalid);
    REQUIRE(!result.clamped);
    REQUIRE_NEAR(result.radiance.length_squared(), 0.0, 1e-12);
}

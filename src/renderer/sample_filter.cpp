#include "sample_filter.h"

#include <algorithm>
#include <cmath>

namespace {

bool finite_color(const color &value) {
    return std::isfinite(value.x()) && std::isfinite(value.y()) &&
           std::isfinite(value.z());
}

double luminance(const color &value) {
    return 0.2126 * value.x() + 0.7152 * value.y() + 0.0722 * value.z();
}

} // namespace

FilteredCameraSample filter_camera_sample(const color &radiance,
                                          double luminance_limit) {
    FilteredCameraSample result;
    if (!finite_color(radiance)) {
        result.invalid = true;
        return result;
    }

    result.radiance = radiance;
    const double sample_luminance = luminance(radiance);
    if (luminance_limit > 0.0 && std::isfinite(sample_luminance) &&
        sample_luminance > luminance_limit) {
        result.radiance *= luminance_limit / sample_luminance;
        result.clamped = true;
    }
    return result;
}

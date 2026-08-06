#include "camera_config.h"

#include <cmath>
#include <stdexcept>

namespace {

bool finite(const vec3 &value) {
    return std::isfinite(value.x()) && std::isfinite(value.y()) &&
           std::isfinite(value.z());
}

} // namespace

void validate_camera_config(const CameraConfig &camera) {
    if (!finite(camera.lookfrom) || !finite(camera.lookat) ||
        !finite(camera.vup)) {
        throw std::invalid_argument("camera vectors must be finite");
    }
    const vec3 view = camera.lookat - camera.lookfrom;
    if (view.near_zero()) {
        throw std::invalid_argument(
            "camera lookfrom and lookat must be distinct");
    }
    if (camera.vup.near_zero() || cross(camera.vup, view).near_zero()) {
        throw std::invalid_argument(
            "camera up vector must define a valid frame");
    }
    if (!std::isfinite(camera.vfov) || camera.vfov <= 0.0 ||
        camera.vfov >= 180.0) {
        throw std::invalid_argument(
            "camera vertical field of view must be in (0, 180)");
    }
    if (!std::isfinite(camera.aperture) || camera.aperture < 0.0) {
        throw std::invalid_argument(
            "camera aperture must be finite and non-negative");
    }
    if (!std::isfinite(camera.focus_dist) || camera.focus_dist <= 0.0) {
        throw std::invalid_argument(
            "camera focus distance must be positive and finite");
    }
    if (!std::isfinite(camera.aspect_ratio) ||
        camera.aspect_ratio <= 0.0) {
        throw std::invalid_argument(
            "camera aspect ratio must be positive and finite");
    }
}

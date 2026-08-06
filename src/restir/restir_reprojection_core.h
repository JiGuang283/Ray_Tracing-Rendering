#ifndef RESTIR_REPROJECTION_CORE_H
#define RESTIR_REPROJECTION_CORE_H

#include "restir_spatial_core.h"

namespace restir {

enum class RestirTemporalRejection : std::uint32_t {
    Accepted = 0,
    NoHistory = 1,
    InvalidCurrent = 2,
    ProjectionFailure = 3,
    OutsideFrame = 4,
    InvalidPrevious = 5,
    UnsupportedDomain = 6,
    SurfaceDomainMismatch = 7,
    GeometryMismatch = 8,
    MaterialMismatch = 9,
    DepthMismatch = 10,
    NormalMismatch = 11,
    InvalidReservoir = 12,
    AgeLimit = 13,
    NonFinite = 14,
    DestinationOutsideSupport = 15,
    Count = 16,
};

struct RestirReprojection {
    std::uint32_t previous_pixel = kInvalidPackedIndex;
    float expected_previous_depth = 0.0f;
    float previous_u = 0.0f;
    float previous_v = 0.0f;

    RT_HOST_DEVICE bool valid() const noexcept {
        return previous_pixel != kInvalidPackedIndex;
    }
};

RT_HOST_DEVICE RT_FORCE_INLINE RestirTemporalRejection
reproject_restir_surface(const PackedCamera &previous_camera,
                         const RestirSurface &current,
                         std::uint32_t width, std::uint32_t height,
                         RestirReprojection &result) noexcept {
    using namespace packed_transport::math;
    result = {};
    result.previous_pixel = kInvalidPackedIndex;
    if (!current.valid()) {
        return RestirTemporalRejection::InvalidCurrent;
    }
    if (width < 2u || height < 2u || !finite(current.position)) {
        return RestirTemporalRejection::NonFinite;
    }
    const Float3 image_center =
        add(previous_camera.lower_left_corner,
            add(multiply(previous_camera.horizontal, 0.5f),
                multiply(previous_camera.vertical, 0.5f)));
    const Float3 forward = normalize(
        add(image_center, multiply(previous_camera.origin, -1.0f)));
    const Float3 relative =
        add(current.position, multiply(previous_camera.origin, -1.0f));
    const float depth = dot(relative, forward);
    const float plane_distance = dot(
        add(image_center, multiply(previous_camera.origin, -1.0f)),
        forward);
    const float horizontal_length_squared =
        dot(previous_camera.horizontal, previous_camera.horizontal);
    const float vertical_length_squared =
        dot(previous_camera.vertical, previous_camera.vertical);
    if (!finite(forward) || !finite(depth) || !finite(plane_distance) ||
        !finite(horizontal_length_squared) ||
        !finite(vertical_length_squared) || !(depth > 1.0e-7f) ||
        !(plane_distance > 1.0e-7f) ||
        !(horizontal_length_squared > 1.0e-12f) ||
        !(vertical_length_squared > 1.0e-12f)) {
        return RestirTemporalRejection::ProjectionFailure;
    }
    const Float3 projected = add(
        previous_camera.origin,
        multiply(relative, plane_distance / depth));
    const Float3 plane_offset = add(
        projected,
        multiply(previous_camera.lower_left_corner, -1.0f));
    const float u = dot(plane_offset, previous_camera.horizontal) /
                    horizontal_length_squared;
    const float v = dot(plane_offset, previous_camera.vertical) /
                    vertical_length_squared;
    const float pixel_x = u * static_cast<float>(width - 1u);
    const float pixel_y = v * static_cast<float>(height - 1u);
    if (!finite(u) || !finite(v) || !finite(pixel_x) || !finite(pixel_y)) {
        return RestirTemporalRejection::NonFinite;
    }
    if (pixel_x < 0.0f || pixel_y < 0.0f ||
        pixel_x >= static_cast<float>(width) ||
        pixel_y >= static_cast<float>(height)) {
        return RestirTemporalRejection::OutsideFrame;
    }
    const std::uint32_t x = static_cast<std::uint32_t>(pixel_x);
    const std::uint32_t y = static_cast<std::uint32_t>(pixel_y);
    result.previous_pixel = y * width + x;
    result.expected_previous_depth = depth;
    result.previous_u = u;
    result.previous_v = v;
    return RestirTemporalRejection::Accepted;
}

RT_HOST_DEVICE RT_FORCE_INLINE RestirTemporalRejection
restir_temporal_compatibility(
    const RestirSurface &current, const RestirSurface &previous,
    float expected_previous_depth, float normal_threshold,
    float depth_threshold) noexcept {
    if (!current.valid()) {
        return RestirTemporalRejection::InvalidCurrent;
    }
    if (!previous.valid()) {
        return RestirTemporalRejection::InvalidPrevious;
    }
    constexpr std::uint32_t kUnsupported =
        RESTIR_SURFACE_UNSUPPORTED_DOMAIN | RESTIR_SURFACE_NO_SCATTER;
    if ((current.flags & kUnsupported) != 0u ||
        (previous.flags & kUnsupported) != 0u) {
        return RestirTemporalRejection::UnsupportedDomain;
    }
    if (current.delta_only() != previous.delta_only()) {
        return RestirTemporalRejection::SurfaceDomainMismatch;
    }
    if (current.instance_id != previous.instance_id) {
        return RestirTemporalRejection::GeometryMismatch;
    }
    if (current.material_id != previous.material_id) {
        return RestirTemporalRejection::MaterialMismatch;
    }
    if (!surface_detail::finite(expected_previous_depth) ||
        !surface_detail::finite(previous.view_depth) ||
        !surface_detail::finite(normal_threshold) ||
        !surface_detail::finite(depth_threshold)) {
        return RestirTemporalRejection::NonFinite;
    }
    const float expected =
        surface_detail::absolute(expected_previous_depth);
    const float stored = surface_detail::absolute(previous.view_depth);
    const float depth_scale = expected > stored ? expected : stored;
    const float stable_scale = depth_scale > 1.0e-4f ? depth_scale : 1.0e-4f;
    if (surface_detail::absolute(expected_previous_depth -
                                 previous.view_depth) >
        depth_threshold * stable_scale) {
        return RestirTemporalRejection::DepthMismatch;
    }
    const Float3 current_normal =
        unpack_octahedral_normal(current.shading_normal);
    const Float3 previous_normal =
        unpack_octahedral_normal(previous.shading_normal);
    const float normal_dot = current_normal.x * previous_normal.x +
                             current_normal.y * previous_normal.y +
                             current_normal.z * previous_normal.z;
    if (!surface_detail::finite(normal_dot)) {
        return RestirTemporalRejection::NonFinite;
    }
    return normal_dot >= normal_threshold
               ? RestirTemporalRejection::Accepted
               : RestirTemporalRejection::NormalMismatch;
}

} // namespace restir

#endif

#ifndef RESTIR_SPATIAL_CORE_H
#define RESTIR_SPATIAL_CORE_H

#include "restir_di_core.h"

#include <cstdint>

namespace restir {

constexpr std::uint32_t kMaxRestirSpatialNeighbors = 64u;

enum class RestirSpatialCompatibility : std::uint32_t {
    Compatible = 0,
    InvalidCenter = 1,
    InvalidNeighbor = 2,
    UnsupportedDomain = 3,
    SurfaceDomainMismatch = 4,
    MaterialMismatch = 5,
    DepthMismatch = 6,
    NormalMismatch = 7,
    NonFinite = 8,
    Count = 9,
};

struct RestirSpatialNeighbor {
    std::uint32_t pixel = 0;
    std::int32_t offset_x = 0;
    std::int32_t offset_y = 0;
    std::uint32_t valid = 0;
};

RT_HOST_DEVICE RT_FORCE_INLINE RestirSpatialNeighbor restir_spatial_neighbor(
    std::uint32_t center_pixel, std::uint32_t ordinal,
    std::uint32_t width, std::uint32_t height, std::uint32_t iteration,
    std::uint32_t pass_index, std::uint32_t seed) noexcept {
    RestirSpatialNeighbor result;
    if (width == 0u || height == 0u || ordinal >= kMaxRestirSpatialNeighbors ||
        center_pixel >= static_cast<std::uint64_t>(width) * height) {
        return result;
    }

    constexpr std::uint32_t kNeighborDomain = 0x53504154u;
    const std::uint32_t hash = restir_random_seed(
        seed ^ mix_seed(iteration, pass_index), center_pixel, pass_index,
        kNeighborDomain);
    const std::uint32_t stride = ((hash >> 8u) & 31u) * 2u + 1u;
    const std::uint32_t slot = ((hash & 63u) + ordinal * stride) & 63u;
    std::int32_t offset_x = static_cast<std::int32_t>(slot & 7u) - 4;
    std::int32_t offset_y = static_cast<std::int32_t>(slot >> 3u) - 4;
    // The 8x8 lattice contains the center but not (+4,+4), so this mapping
    // preserves uniqueness while excluding self-reuse.
    if (offset_x == 0 && offset_y == 0) {
        offset_x = 4;
        offset_y = 4;
    }
    if ((hash & 0x4000u) != 0u) {
        const std::int32_t temporary = offset_x;
        offset_x = offset_y;
        offset_y = temporary;
    }
    if ((hash & 0x8000u) != 0u) {
        offset_x = -offset_x;
    }
    if ((hash & 0x10000u) != 0u) {
        offset_y = -offset_y;
    }

    result.offset_x = offset_x;
    result.offset_y = offset_y;
    const std::int32_t center_x =
        static_cast<std::int32_t>(center_pixel % width);
    const std::int32_t center_y =
        static_cast<std::int32_t>(center_pixel / width);
    const std::int32_t neighbor_x = center_x + offset_x;
    const std::int32_t neighbor_y = center_y + offset_y;
    if (neighbor_x < 0 || neighbor_y < 0 ||
        neighbor_x >= static_cast<std::int32_t>(width) ||
        neighbor_y >= static_cast<std::int32_t>(height)) {
        return result;
    }
    result.pixel = static_cast<std::uint32_t>(neighbor_y) * width +
                   static_cast<std::uint32_t>(neighbor_x);
    result.valid = 1u;
    return result;
}

RT_HOST_DEVICE RT_FORCE_INLINE RestirSpatialCompatibility
restir_spatial_compatibility(const RestirSurface &center,
                             const RestirSurface &neighbor,
                             float normal_threshold,
                             float depth_threshold) noexcept {
    if (!center.valid()) {
        return RestirSpatialCompatibility::InvalidCenter;
    }
    if (!neighbor.valid()) {
        return RestirSpatialCompatibility::InvalidNeighbor;
    }
    constexpr std::uint32_t kUnsupported =
        RESTIR_SURFACE_UNSUPPORTED_DOMAIN | RESTIR_SURFACE_NO_SCATTER;
    if ((center.flags & kUnsupported) != 0u ||
        (neighbor.flags & kUnsupported) != 0u) {
        return RestirSpatialCompatibility::UnsupportedDomain;
    }
    if (center.delta_only() || neighbor.delta_only() ||
        ((center.flags ^ neighbor.flags) & RESTIR_SURFACE_DELTA_ONLY) != 0u) {
        return RestirSpatialCompatibility::SurfaceDomainMismatch;
    }
    if (center.material_id != neighbor.material_id) {
        return RestirSpatialCompatibility::MaterialMismatch;
    }
    if (!surface_detail::finite(center.view_depth) ||
        !surface_detail::finite(neighbor.view_depth) ||
        !surface_detail::finite(normal_threshold) ||
        !surface_detail::finite(depth_threshold)) {
        return RestirSpatialCompatibility::NonFinite;
    }
    const float center_depth =
        surface_detail::absolute(center.view_depth);
    const float neighbor_depth =
        surface_detail::absolute(neighbor.view_depth);
    const float depth_scale =
        center_depth > neighbor_depth ? center_depth : neighbor_depth;
    const float stable_scale = depth_scale > 1.0e-4f ? depth_scale : 1.0e-4f;
    if (surface_detail::absolute(center.view_depth - neighbor.view_depth) >
        depth_threshold * stable_scale) {
        return RestirSpatialCompatibility::DepthMismatch;
    }
    const Float3 center_normal =
        unpack_octahedral_normal(center.shading_normal);
    const Float3 neighbor_normal =
        unpack_octahedral_normal(neighbor.shading_normal);
    const float normal_dot = center_normal.x * neighbor_normal.x +
                             center_normal.y * neighbor_normal.y +
                             center_normal.z * neighbor_normal.z;
    if (!surface_detail::finite(normal_dot)) {
        return RestirSpatialCompatibility::NonFinite;
    }
    if (normal_dot < normal_threshold) {
        return RestirSpatialCompatibility::NormalMismatch;
    }
    return RestirSpatialCompatibility::Compatible;
}

} // namespace restir

#endif

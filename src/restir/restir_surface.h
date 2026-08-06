#ifndef RESTIR_SURFACE_H
#define RESTIR_SURFACE_H

#include "packed_types.h"

#include <cmath>
#include <cstdint>
#include <type_traits>

namespace restir {

enum RestirSurfaceFlagBits : std::uint32_t {
    RESTIR_SURFACE_NONE = 0,
    RESTIR_SURFACE_HIT_FLAGS_MASK = 0xffu,
    RESTIR_SURFACE_VALID = 1u << 8u,
    RESTIR_SURFACE_DELTA_ONLY = 1u << 9u,
    RESTIR_SURFACE_NO_SCATTER = 1u << 10u,
    RESTIR_SURFACE_UNSUPPORTED_DOMAIN = 1u << 11u,
};

enum class RestirGBufferStatus : std::uint32_t {
    Success = 0,
    Miss = 1,
    InvalidInput = 2,
    TraversalFailure = 3,
    ReconstructionFailure = 4,
    MaterialFailure = 5,
    NonFinite = 6,
};

struct alignas(16) RestirSurface {
    Float3 position{};
    float view_depth = 0.0f;

    float hit_t = 0.0f;
    float barycentric_u = 0.0f;
    float barycentric_v = 0.0f;
    float ray_time = 0.0f;

    std::uint32_t instance_id = kInvalidPackedIndex;
    std::uint32_t primitive_id = kInvalidPackedIndex;
    std::uint32_t material_id = kInvalidPackedIndex;
    std::uint32_t emitter_id = kInvalidPackedIndex;

    std::uint32_t geometric_normal = 0;
    std::uint32_t shading_normal = 0;
    std::uint32_t flags = RESTIR_SURFACE_NONE;
    // Reserved for a packed current-to-previous pixel motion vector.
    std::uint32_t motion = 0;

    RT_HOST_DEVICE bool valid() const noexcept {
        return (flags & RESTIR_SURFACE_VALID) != 0u;
    }

    RT_HOST_DEVICE bool delta_only() const noexcept {
        return (flags & RESTIR_SURFACE_DELTA_ONLY) != 0u;
    }
};

namespace surface_detail {

RT_HOST_DEVICE RT_FORCE_INLINE float absolute(float value) noexcept {
    return value < 0.0f ? -value : value;
}

RT_HOST_DEVICE RT_FORCE_INLINE float clamp_unit(float value) noexcept {
    return value < -1.0f ? -1.0f : (value > 1.0f ? 1.0f : value);
}

RT_HOST_DEVICE RT_FORCE_INLINE bool finite(float value) noexcept {
    return value == value && value <= 3.402823466e+38f &&
           value >= -3.402823466e+38f;
}

RT_HOST_DEVICE RT_FORCE_INLINE bool finite(Float3 value) noexcept {
    return finite(value.x) && finite(value.y) && finite(value.z);
}

RT_HOST_DEVICE RT_FORCE_INLINE float sign_not_zero(float value) noexcept {
    return value < 0.0f ? -1.0f : 1.0f;
}

RT_HOST_DEVICE RT_FORCE_INLINE std::int32_t quantize_snorm16(
    float value) noexcept {
    const float scaled = clamp_unit(value) * 32767.0f;
    return static_cast<std::int32_t>(
        scaled + (scaled < 0.0f ? -0.5f : 0.5f));
}

} // namespace surface_detail

RT_HOST_DEVICE RT_FORCE_INLINE std::uint32_t
pack_octahedral_normal(Float3 normal) noexcept {
    if (!surface_detail::finite(normal)) {
        return 0u;
    }
    const float length_squared =
        normal.x * normal.x + normal.y * normal.y + normal.z * normal.z;
    if (!(length_squared > 0.0f)) {
        return 0u;
    }
    const float inverse_length = 1.0f / ::sqrtf(length_squared);
    normal.x *= inverse_length;
    normal.y *= inverse_length;
    normal.z *= inverse_length;
    const float inverse_l1 =
        1.0f / (surface_detail::absolute(normal.x) +
                surface_detail::absolute(normal.y) +
                surface_detail::absolute(normal.z));
    float x = normal.x * inverse_l1;
    float y = normal.y * inverse_l1;
    if (normal.z < 0.0f) {
        const float old_x = x;
        x = (1.0f - surface_detail::absolute(y)) *
            surface_detail::sign_not_zero(old_x);
        y = (1.0f - surface_detail::absolute(old_x)) *
            surface_detail::sign_not_zero(y);
    }
    const std::int16_t packed_x = static_cast<std::int16_t>(
        surface_detail::quantize_snorm16(x));
    const std::int16_t packed_y = static_cast<std::int16_t>(
        surface_detail::quantize_snorm16(y));
    return static_cast<std::uint16_t>(packed_x) |
           (static_cast<std::uint32_t>(
                static_cast<std::uint16_t>(packed_y))
            << 16u);
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3
unpack_octahedral_normal(std::uint32_t packed) noexcept {
    const std::int16_t packed_x =
        static_cast<std::int16_t>(packed & 0xffffu);
    const std::int16_t packed_y =
        static_cast<std::int16_t>((packed >> 16u) & 0xffffu);
    float x = static_cast<float>(packed_x) / 32767.0f;
    float y = static_cast<float>(packed_y) / 32767.0f;
    float z = 1.0f - surface_detail::absolute(x) -
              surface_detail::absolute(y);
    if (z < 0.0f) {
        const float old_x = x;
        x = (1.0f - surface_detail::absolute(y)) *
            surface_detail::sign_not_zero(old_x);
        y = (1.0f - surface_detail::absolute(old_x)) *
            surface_detail::sign_not_zero(y);
    }
    const float length_squared = x * x + y * y + z * z;
    if (!(length_squared > 0.0f)) {
        return {0.0f, 0.0f, 1.0f};
    }
    const float inverse_length = 1.0f / ::sqrtf(length_squared);
    return {x * inverse_length, y * inverse_length, z * inverse_length};
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedHit
restir_surface_hit(const RestirSurface &surface) noexcept {
    PackedHit hit;
    hit.t = surface.hit_t;
    hit.barycentric_u = surface.barycentric_u;
    hit.barycentric_v = surface.barycentric_v;
    hit.instance_id = surface.instance_id;
    hit.primitive_id = surface.primitive_id;
    hit.flags = surface.flags & RESTIR_SURFACE_HIT_FLAGS_MASK;
    return hit;
}

static_assert(sizeof(RestirSurface) == 64,
              "RestirSurface must remain a compact 64-byte GBuffer record");
static_assert(alignof(RestirSurface) == 16);
static_assert(std::is_trivially_copyable_v<RestirSurface>);
static_assert(std::is_trivially_copyable_v<RestirGBufferStatus>);

} // namespace restir

#endif

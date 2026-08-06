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

struct RestirDISpatialStats {
    std::uint32_t candidates = 0;
    std::uint32_t accepted = 0;
    std::uint32_t rejected = 0;
    std::uint32_t pairwise_fallbacks = 0;
    std::uint32_t compatibility[
        static_cast<std::uint32_t>(RestirSpatialCompatibility::Count)]{};
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

RT_HOST_DEVICE RT_FORCE_INLINE RestirDIStatus
evaluate_restir_sample_target(const CompiledSceneView &scene,
                              const RestirDIPixelContext &context,
                              const RestirLightSample &sample,
                              float &target) noexcept {
    target = 0.0f;
    PackedLightSample evaluated;
    const PackedLightStatus light_status =
        evaluate_restir_light_sample_core(
            scene, sample, context.surface.position, evaluated);
    if (light_status == PackedLightStatus::NoSample) {
        return RestirDIStatus::Success;
    }
    if (light_status != PackedLightStatus::Success) {
        return RestirDIStatus::LightFailure;
    }
    Float3 integrand{};
    return evaluate_unshadowed_restir_direct(context, evaluated, integrand,
                                              target);
}

RT_HOST_DEVICE RT_FORCE_INLINE void capped_di_source_mass(
    const RestirDIReservoir &source, std::uint32_t current_M,
    std::uint32_t max_candidates, std::uint32_t &represented_count,
    float &effective_count, float &mass_fraction) noexcept {
    specialized_reservoir::capped_source_mass(
        source, current_M, max_candidates, represented_count,
        effective_count, mass_fraction);
}

RT_HOST_DEVICE RT_FORCE_INLINE RestirDIStatus combine_basic_di_source(
    const CompiledSceneView &scene, const RestirDIPixelContext &destination,
    const RestirDIReservoir &source, std::uint32_t max_candidates,
    float random, RestirDIReservoir &output, bool &accepted,
    bool *changed_selection = nullptr) noexcept {
    accepted = false;
    if (changed_selection != nullptr) {
        *changed_selection = false;
    }
    std::uint32_t represented_count = 0u;
    float effective_count = 0.0f;
    float mass_fraction = 0.0f;
    capped_di_source_mass(source, output.M, max_candidates,
                          represented_count, effective_count, mass_fraction);
    if (represented_count == 0u) {
        return RestirDIStatus::Success;
    }
    if (!reservoir_is_usable(source)) {
        const ReservoirOperationResult represented =
            represent_di_candidates(output, represented_count,
                                    effective_count);
        return represented.accepted() ||
                       represented.rejection ==
                           ReservoirRejectReason::ZeroTarget
                   ? RestirDIStatus::Success
                   : RestirDIStatus::ReservoirFailure;
    }

    float destination_target = 0.0f;
    const RestirDIStatus target_status = evaluate_restir_sample_target(
        scene, destination, source.sample, destination_target);
    if (target_status != RestirDIStatus::Success) {
        return target_status;
    }
    if (!(destination_target > 0.0f)) {
        const ReservoirOperationResult represented =
            represent_di_candidates(output, represented_count,
                                    effective_count);
        return represented.accepted() ||
                       represented.rejection ==
                           ReservoirRejectReason::ZeroTarget
                   ? RestirDIStatus::Success
                   : RestirDIStatus::ReservoirFailure;
    }
    const float stream_normalization =
        source.unbiased_contribution_weight * source.effective_M *
        mass_fraction;
    const float weight = destination_target * stream_normalization;
    const ReservoirOperationResult combined = stream_di_weight(
        output, source.sample, destination_target, weight,
        represented_count, effective_count, random);
    if (!combined.accepted()) {
        return RestirDIStatus::ReservoirFailure;
    }
    accepted = true;
    if (changed_selection != nullptr) {
        *changed_selection = combined.changed_selection();
    }
    return RestirDIStatus::Success;
}

RT_HOST_DEVICE RT_FORCE_INLINE RestirDIStatus spatial_resample_di_basic(
    const CompiledSceneView &scene, const RestirSurface *surfaces,
    const RestirDIReservoir *source_reservoirs, std::uint32_t width,
    std::uint32_t height, std::uint32_t pixel, std::uint32_t iteration,
    std::uint32_t pass_index, std::uint32_t seed,
    std::uint32_t neighbor_count, std::uint32_t max_candidates,
    float normal_threshold, float depth_threshold,
    RestirDIReservoir &output, RestirDISpatialStats &stats) noexcept {
    reset_reservoir(output);
    stats = {};
    if (surfaces == nullptr || source_reservoirs == nullptr || width < 2u ||
        height < 2u ||
        pixel >= static_cast<std::uint64_t>(width) * height ||
        neighbor_count > kMaxRestirSpatialNeighbors) {
        return RestirDIStatus::ReservoirFailure;
    }
    RestirDIPixelContext destination;
    const RestirDIStatus context_status = reconstruct_restir_di_context(
        scene, surfaces[pixel], width, height, pixel, iteration, seed,
        destination);
    if (context_status != RestirDIStatus::Success) {
        return context_status;
    }
    constexpr std::uint32_t kSpatialReservoirDomain = 0x53505253u;
    RNG rng(restir_random_seed(seed, pixel, iteration,
                              kSpatialReservoirDomain ^ pass_index));

    ++stats.candidates;
    bool accepted = false;
    bool changed_selection = false;
    RestirDIStatus status = combine_basic_di_source(
        scene, destination, source_reservoirs[pixel], max_candidates,
        static_cast<float>(rng.next()), output, accepted,
        &changed_selection);
    if (status != RestirDIStatus::Success) {
        ++stats.rejected;
        return status;
    }
    if (changed_selection) {
        output.age = source_reservoirs[pixel].age;
    }
    stats.accepted += accepted ? 1u : 0u;

    for (std::uint32_t ordinal = 0; ordinal < neighbor_count; ++ordinal) {
        if (max_candidates != 0u && output.M >= max_candidates) {
            break;
        }
        const RestirSpatialNeighbor neighbor = restir_spatial_neighbor(
            pixel, ordinal, width, height, iteration, pass_index, seed);
        if (neighbor.valid == 0u) {
            continue;
        }
        ++stats.candidates;
        const RestirSpatialCompatibility compatibility =
            restir_spatial_compatibility(
                surfaces[pixel], surfaces[neighbor.pixel], normal_threshold,
                depth_threshold);
        ++stats.compatibility[static_cast<std::uint32_t>(compatibility)];
        if (compatibility != RestirSpatialCompatibility::Compatible) {
            ++stats.rejected;
            continue;
        }
        accepted = false;
        changed_selection = false;
        status = combine_basic_di_source(
            scene, destination, source_reservoirs[neighbor.pixel],
            max_candidates, static_cast<float>(rng.next()), output,
            accepted, &changed_selection);
        if (status != RestirDIStatus::Success) {
            ++stats.rejected;
            return status;
        }
        if (changed_selection) {
            output.age = source_reservoirs[neighbor.pixel].age;
        }
        stats.accepted += accepted ? 1u : 0u;
    }
    const ReservoirOperationResult finalized = finalize_reservoir(output);
    if (!finalized.accepted()) {
        return finalized.rejection == ReservoirRejectReason::EmptyReservoir ||
                       finalized.rejection ==
                           ReservoirRejectReason::NoSelectedSample
                   ? RestirDIStatus::ReservoirEmpty
                   : RestirDIStatus::ReservoirFailure;
    }
    return RestirDIStatus::Success;
}

} // namespace restir

#endif

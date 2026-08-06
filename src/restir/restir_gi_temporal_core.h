#ifndef RESTIR_GI_TEMPORAL_CORE_H
#define RESTIR_GI_TEMPORAL_CORE_H

#include "restir_gi_pairwise_core.h"
#include "restir_gi_spatial_core.h"
#include "restir_reprojection_core.h"

namespace restir {

struct RestirGITemporalStats {
    std::uint32_t candidates = 0u;
    std::uint32_t accepted = 0u;
    RestirTemporalRejection rejection =
        RestirTemporalRejection::NoHistory;
    RestirGIShiftFailure shift_failure = RestirGIShiftFailure::None;
    std::uint32_t pairwise_fallbacks = 0u;
};

RT_HOST_DEVICE RT_FORCE_INLINE bool prepare_gi_temporal_candidate(
    const PackedCamera *previous_camera, RestirSurface &current_surface,
    const RestirSurface *previous_surfaces,
    const RestirGIReservoir *previous_reservoirs, std::uint32_t width,
    std::uint32_t height, std::uint32_t max_history_length,
    float normal_threshold, float depth_threshold,
    RestirReprojection &projection, RestirGITemporalStats &stats) noexcept {
    stats.candidates = 1u;
    if (previous_camera == nullptr || previous_surfaces == nullptr ||
        previous_reservoirs == nullptr) {
        return false;
    }
    stats.rejection = reproject_restir_surface(
        *previous_camera, current_surface, width, height, projection);
    if (stats.rejection != RestirTemporalRejection::Accepted) {
        return false;
    }
    current_surface.motion = projection.previous_pixel;
    const RestirSurface &previous_surface =
        previous_surfaces[projection.previous_pixel];
    stats.rejection = restir_temporal_compatibility(
        current_surface, previous_surface,
        projection.expected_previous_depth, normal_threshold,
        depth_threshold);
    if (stats.rejection != RestirTemporalRejection::Accepted) {
        return false;
    }
    const RestirGIReservoir &previous =
        previous_reservoirs[projection.previous_pixel];
    if (!reservoir_is_usable(previous)) {
        stats.rejection = RestirTemporalRejection::InvalidReservoir;
        return false;
    }
    if (previous.age >= max_history_length) {
        stats.rejection = RestirTemporalRejection::AgeLimit;
        return false;
    }
    return true;
}

RT_HOST_DEVICE RT_FORCE_INLINE RestirGIStatus temporal_resample_gi_basic(
    const CompiledSceneView &scene, RestirSurface &current_surface,
    const RestirGIReservoir &current_reservoir,
    const PackedCamera *previous_camera,
    const RestirSurface *previous_surfaces,
    const RestirGIReservoir *previous_reservoirs,
    std::uint32_t width, std::uint32_t height, std::uint32_t pixel,
    std::uint32_t iteration, std::uint32_t seed,
    std::uint32_t max_history_length, std::uint32_t max_candidates,
    float normal_threshold, float depth_threshold,
    RestirGIReservoir &output,
    RestirGITemporalStats &stats) noexcept {
    output = current_reservoir;
    stats = {};
    RestirReprojection projection;
    if (!prepare_gi_temporal_candidate(
            previous_camera, current_surface, previous_surfaces,
            previous_reservoirs, width, height, max_history_length,
            normal_threshold, depth_threshold, projection, stats)) {
        return RestirGIStatus::Success;
    }
    const RestirGIReservoir &previous =
        previous_reservoirs[projection.previous_pixel];
    RestirDIPixelContext destination;
    const RestirGIStatus context_status = reconstruct_restir_gi_context(
        scene, current_surface, width, height, pixel, iteration, seed,
        destination);
    if (context_status != RestirGIStatus::Success) {
        stats.rejection = RestirTemporalRejection::NonFinite;
        return context_status;
    }

    reset_reservoir(output);
    constexpr std::uint32_t kTemporalDomain = 0x4749544du;
    RNG rng(restir_random_seed(seed, pixel, iteration, kTemporalDomain));
    bool current_accepted = false;
    bool current_selected = false;
    RestirGIStatus status = combine_basic_gi_source(
        destination, current_reservoir, max_candidates,
        static_cast<float>(rng.next()), output, current_accepted,
        &current_selected);
    if (status != RestirGIStatus::Success) {
        stats.rejection = RestirTemporalRejection::NonFinite;
        return status;
    }
    bool history_accepted = false;
    bool history_selected = false;
    status = combine_basic_gi_source(
        destination, previous, max_candidates,
        static_cast<float>(rng.next()), output, history_accepted,
        &history_selected, &stats.shift_failure);
    if (status != RestirGIStatus::Success) {
        stats.rejection = RestirTemporalRejection::NonFinite;
        return status;
    }
    if (history_accepted) {
        stats.accepted = 1u;
        stats.rejection = RestirTemporalRejection::Accepted;
    } else {
        stats.rejection =
            RestirTemporalRejection::DestinationOutsideSupport;
    }
    const ReservoirOperationResult finalized = finalize_reservoir(output);
    if (!finalized.accepted()) {
        return finalized.rejection == ReservoirRejectReason::EmptyReservoir ||
                       finalized.rejection ==
                           ReservoirRejectReason::NoSelectedSample
                   ? RestirGIStatus::ReservoirEmpty
                   : RestirGIStatus::ReservoirFailure;
    }
    output.age = history_selected ? previous.age + 1u : 0u;
    return RestirGIStatus::Success;
}

RT_HOST_DEVICE RT_FORCE_INLINE RestirGIStatus temporal_resample_gi_pairwise(
    const CompiledSceneView &scene, RestirSurface &current_surface,
    const RestirGIReservoir &current_reservoir,
    const PackedCamera *previous_camera,
    const RestirSurface *previous_surfaces,
    const RestirGIReservoir *previous_reservoirs, std::uint32_t width,
    std::uint32_t height, std::uint32_t pixel, std::uint32_t iteration,
    std::uint32_t seed, std::uint32_t max_history_length,
    std::uint32_t max_candidates, float normal_threshold,
    float depth_threshold, RestirGIReservoir &output,
    RestirGITemporalStats &stats) noexcept {
    if (!reservoir_is_usable(current_reservoir)) {
        const RestirGIStatus fallback = temporal_resample_gi_basic(
            scene, current_surface, current_reservoir, previous_camera,
            previous_surfaces, previous_reservoirs, width, height, pixel,
            iteration, seed, max_history_length, max_candidates,
            normal_threshold, depth_threshold, output, stats);
        ++stats.pairwise_fallbacks;
        return fallback;
    }
    output = current_reservoir;
    stats = {};
    RestirReprojection projection;
    if (!prepare_gi_temporal_candidate(
            previous_camera, current_surface, previous_surfaces,
            previous_reservoirs, width, height, max_history_length,
            normal_threshold, depth_threshold, projection, stats)) {
        return RestirGIStatus::Success;
    }
    RestirDIPixelContext current_context;
    RestirGIStatus status = reconstruct_restir_gi_context(
        scene, current_surface, width, height, pixel, iteration, seed,
        current_context);
    if (status != RestirGIStatus::Success) {
        return status;
    }
    const std::uint32_t previous_iteration =
        iteration > 0u ? iteration - 1u : 0u;
    RestirDIPixelContext previous_context;
    status = reconstruct_restir_gi_context(
        scene, previous_surfaces[projection.previous_pixel],
        *previous_camera, width, height, projection.previous_pixel,
        previous_iteration, seed, previous_context);
    if (status != RestirGIStatus::Success) {
        return status;
    }
    const RestirGIReservoir &previous =
        previous_reservoirs[projection.previous_pixel];
    std::uint32_t current_count = 0u;
    float current_effective = 0.0f;
    float ignored_fraction = 0.0f;
    specialized_reservoir::capped_source_mass(
        current_reservoir, 0u, max_candidates, current_count,
        current_effective, ignored_fraction);
    std::uint32_t previous_count = 0u;
    float previous_effective = 0.0f;
    specialized_reservoir::capped_source_mass(
        previous, current_count, max_candidates, previous_count,
        previous_effective, ignored_fraction);
    if (previous_count == 0u) {
        stats.rejection = RestirTemporalRejection::InvalidReservoir;
        return RestirGIStatus::Success;
    }
    float previous_at_current = 0.0f;
    float previous_at_previous = 0.0f;
    float current_at_previous = 0.0f;
    float current_at_current = 0.0f;
    RestirGIShiftFailure failure = RestirGIShiftFailure::None;
    status = evaluate_gi_pairwise_target(
        current_context, previous.sample, previous_at_current, failure);
    stats.shift_failure = failure;
    if (status == RestirGIStatus::Success) {
        status = evaluate_gi_pairwise_target(
            previous_context, previous.sample, previous_at_previous,
            failure);
    }
    if (status == RestirGIStatus::Success) {
        status = evaluate_gi_pairwise_target(
            previous_context, current_reservoir.sample,
            current_at_previous, failure);
    }
    if (status == RestirGIStatus::Success) {
        status = evaluate_gi_pairwise_target(
            current_context, current_reservoir.sample,
            current_at_current, failure);
    }
    if (failure != RestirGIShiftFailure::None) {
        stats.shift_failure = failure;
    }
    if (status != RestirGIStatus::Success) {
        stats.rejection = RestirTemporalRejection::NonFinite;
        return status;
    }
    const float w_previous = restir_pairwise_mis_weight(
        previous_at_previous, previous_at_current, previous_effective,
        current_effective);
    const float w_current = restir_pairwise_mis_weight(
        current_at_previous, current_at_current, previous_effective,
        current_effective);
    const float previous_factor = restir_pairwise_m_factor(
        previous_at_previous, previous_at_current);
    const float current_factor = restir_pairwise_m_factor(
        current_at_previous, current_at_current);
    const float adjusted_previous = previous_effective *
        (previous_factor < current_factor ? previous_factor
                                          : current_factor);
    reset_reservoir(output);
    constexpr std::uint32_t kTemporalPairwiseDomain = 0x47545057u;
    RNG rng(restir_random_seed(seed, pixel, iteration,
                              kTemporalPairwiseDomain));
    bool selected_history = false;
    const float previous_mass = previous_at_current *
        previous.unbiased_contribution_weight * w_previous;
    if (previous_mass > 0.0f && previous_at_current > 0.0f) {
        const ReservoirOperationResult streamed = stream_gi_weight(
            output, previous.sample, previous_at_current, previous_mass,
            previous_count, adjusted_previous,
            static_cast<float>(rng.next()));
        if (!streamed.accepted()) {
            return RestirGIStatus::ReservoirFailure;
        }
        selected_history = streamed.changed_selection();
        stats.accepted = 1u;
        stats.rejection = RestirTemporalRejection::Accepted;
    } else {
        const ReservoirOperationResult represented =
            represent_gi_candidates(output, previous_count,
                                    adjusted_previous);
        if (represented.rejection != ReservoirRejectReason::ZeroTarget &&
            !represented.accepted()) {
            return RestirGIStatus::ReservoirFailure;
        }
        stats.rejection = RestirTemporalRejection::DestinationOutsideSupport;
    }
    const float current_mass = current_at_current *
        current_reservoir.unbiased_contribution_weight * (1.0f - w_current);
    if (current_mass > 0.0f && current_at_current > 0.0f) {
        const ReservoirOperationResult streamed = stream_gi_weight(
            output, current_reservoir.sample, current_at_current,
            current_mass, current_count, current_effective,
            static_cast<float>(rng.next()));
        if (!streamed.accepted()) {
            return RestirGIStatus::ReservoirFailure;
        }
        if (streamed.changed_selection()) {
            selected_history = false;
        }
    } else {
        const ReservoirOperationResult represented =
            represent_gi_candidates(output, current_count,
                                    current_effective);
        if (represented.rejection != ReservoirRejectReason::ZeroTarget &&
            !represented.accepted()) {
            return RestirGIStatus::ReservoirFailure;
        }
    }
    const ReservoirOperationResult finalized =
        finalize_gi_reservoir(output, 1.0f);
    if (!finalized.accepted()) {
        return RestirGIStatus::ReservoirEmpty;
    }
    output.age = selected_history ? previous.age + 1u : 0u;
    return RestirGIStatus::Success;
}

} // namespace restir

#endif

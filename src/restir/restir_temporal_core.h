#ifndef RESTIR_TEMPORAL_CORE_H
#define RESTIR_TEMPORAL_CORE_H

#include "restir_reprojection_core.h"
#include "restir_spatial_pairwise_core.h"

namespace restir {

struct RestirDITemporalStats {
    std::uint32_t candidates = 0;
    std::uint32_t accepted = 0;
    std::uint32_t pairwise_fallbacks = 0;
    RestirTemporalRejection rejection =
        RestirTemporalRejection::NoHistory;
};

RT_HOST_DEVICE RT_FORCE_INLINE bool prepare_restir_temporal_candidate(
    const PackedCamera *previous_camera,
    RestirSurface &current_surface,
    const RestirSurface *previous_surfaces,
    const RestirDIReservoir *previous_reservoirs,
    std::uint32_t width, std::uint32_t height,
    std::uint32_t max_history_length, float normal_threshold,
    float depth_threshold, RestirReprojection &projection,
    RestirDITemporalStats &stats) noexcept {
    stats.candidates = 1u;
    if (previous_camera == nullptr || previous_surfaces == nullptr ||
        previous_reservoirs == nullptr) {
        stats.rejection = RestirTemporalRejection::NoHistory;
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
    const RestirDIReservoir &previous =
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

RT_HOST_DEVICE RT_FORCE_INLINE RestirDIStatus temporal_resample_di_basic(
    const CompiledSceneView &scene, RestirSurface &current_surface,
    const RestirDIReservoir &current_reservoir,
    const PackedCamera *previous_camera,
    const RestirSurface *previous_surfaces,
    const RestirDIReservoir *previous_reservoirs,
    std::uint32_t width, std::uint32_t height, std::uint32_t pixel,
    std::uint32_t iteration, std::uint32_t seed,
    std::uint32_t max_history_length, std::uint32_t max_candidates,
    float normal_threshold, float depth_threshold,
    RestirDIReservoir &output, RestirDITemporalStats &stats) noexcept {
    output = current_reservoir;
    stats = {};
    RestirReprojection projection;
    if (!prepare_restir_temporal_candidate(
            previous_camera, current_surface, previous_surfaces,
            previous_reservoirs, width, height, max_history_length,
            normal_threshold, depth_threshold, projection, stats)) {
        return RestirDIStatus::Success;
    }
    RestirDIPixelContext current_context;
    const RestirDIStatus context_status = reconstruct_restir_di_context(
        scene, current_surface, width, height, pixel, iteration, seed,
        current_context);
    if (context_status != RestirDIStatus::Success) {
        stats.rejection = RestirTemporalRejection::NonFinite;
        return context_status;
    }

    reset_reservoir(output);
    constexpr std::uint32_t kTemporalDomain = 0x54454d50u;
    RNG rng(restir_random_seed(seed, pixel, iteration, kTemporalDomain));
    bool current_accepted = false;
    bool current_selected = false;
    RestirDIStatus status = combine_basic_di_source(
        scene, current_context, current_reservoir, max_candidates,
        static_cast<float>(rng.next()), output, current_accepted,
        &current_selected);
    if (status != RestirDIStatus::Success) {
        stats.rejection = RestirTemporalRejection::NonFinite;
        return status;
    }
    bool history_accepted = false;
    bool history_selected = false;
    const RestirDIReservoir &previous =
        previous_reservoirs[projection.previous_pixel];
    status = combine_basic_di_source(
        scene, current_context, previous, max_candidates,
        static_cast<float>(rng.next()), output, history_accepted,
        &history_selected);
    if (status != RestirDIStatus::Success) {
        stats.rejection = RestirTemporalRejection::NonFinite;
        return status;
    }
    if (!history_accepted) {
        stats.rejection =
            RestirTemporalRejection::DestinationOutsideSupport;
    } else {
        stats.accepted = 1u;
        stats.rejection = RestirTemporalRejection::Accepted;
    }
    const ReservoirOperationResult finalized = finalize_reservoir(output);
    if (!finalized.accepted()) {
        return finalized.rejection == ReservoirRejectReason::EmptyReservoir ||
                       finalized.rejection ==
                           ReservoirRejectReason::NoSelectedSample
                   ? RestirDIStatus::ReservoirEmpty
                   : RestirDIStatus::ReservoirFailure;
    }
    output.age = history_selected ? previous.age + 1u : 0u;
    return RestirDIStatus::Success;
}

RT_HOST_DEVICE RT_FORCE_INLINE RestirDIStatus temporal_resample_di_pairwise(
    const CompiledSceneView &scene, RestirSurface &current_surface,
    const RestirDIReservoir &current_reservoir,
    const PackedCamera *previous_camera,
    const RestirSurface *previous_surfaces,
    const RestirDIReservoir *previous_reservoirs,
    std::uint32_t width, std::uint32_t height, std::uint32_t pixel,
    std::uint32_t iteration, std::uint32_t seed,
    std::uint32_t max_history_length, std::uint32_t max_candidates,
    float normal_threshold, float depth_threshold,
    RestirDIReservoir &output, RestirDITemporalStats &stats) noexcept {
    if (!reservoir_is_usable(current_reservoir)) {
        const RestirDIStatus fallback = temporal_resample_di_basic(
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
    if (!prepare_restir_temporal_candidate(
            previous_camera, current_surface, previous_surfaces,
            previous_reservoirs, width, height, max_history_length,
            normal_threshold, depth_threshold, projection, stats)) {
        return RestirDIStatus::Success;
    }
    RestirDIPixelContext current_context;
    RestirDIStatus status = reconstruct_restir_di_context(
        scene, current_surface, width, height, pixel, iteration, seed,
        current_context);
    if (status != RestirDIStatus::Success) {
        stats.rejection = RestirTemporalRejection::NonFinite;
        return status;
    }
    const std::uint32_t previous_iteration =
        iteration > 0u ? iteration - 1u : 0u;
    RestirDIPixelContext previous_context;
    status = reconstruct_restir_di_context(
        scene, previous_surfaces[projection.previous_pixel],
        *previous_camera, width, height, projection.previous_pixel,
        previous_iteration, seed, previous_context);
    if (status != RestirDIStatus::Success) {
        stats.rejection = RestirTemporalRejection::NonFinite;
        return status;
    }

    const RestirDIReservoir &previous =
        previous_reservoirs[projection.previous_pixel];
    std::uint32_t current_count = 0u;
    float current_effective_M = 0.0f;
    float current_fraction = 0.0f;
    capped_di_source_mass(current_reservoir, 0u, max_candidates,
                          current_count, current_effective_M,
                          current_fraction);
    std::uint32_t previous_count = 0u;
    float previous_effective_M = 0.0f;
    float previous_fraction = 0.0f;
    capped_di_source_mass(previous, current_count, max_candidates,
                          previous_count, previous_effective_M,
                          previous_fraction);
    if (previous_count == 0u) {
        stats.rejection = RestirTemporalRejection::InvalidReservoir;
        return RestirDIStatus::Success;
    }

    float previous_at_current = 0.0f;
    float previous_at_previous = 0.0f;
    float current_at_previous = 0.0f;
    float current_at_current = 0.0f;
    status = evaluate_restir_sample_target(
        scene, current_context, previous.sample, previous_at_current);
    if (status == RestirDIStatus::Success) {
        status = evaluate_restir_sample_target(
            scene, previous_context, previous.sample,
            previous_at_previous);
    }
    if (status == RestirDIStatus::Success) {
        status = evaluate_restir_sample_target(
            scene, previous_context, current_reservoir.sample,
            current_at_previous);
    }
    if (status == RestirDIStatus::Success) {
        status = evaluate_restir_sample_target(
            scene, current_context, current_reservoir.sample,
            current_at_current);
    }
    if (status != RestirDIStatus::Success) {
        stats.rejection = RestirTemporalRejection::NonFinite;
        return status;
    }

    const float w0 = restir_pairwise_mis_weight(
        previous_at_previous, previous_at_current,
        previous_effective_M, current_effective_M);
    const float w1 = restir_pairwise_mis_weight(
        current_at_previous, current_at_current,
        previous_effective_M, current_effective_M);
    const float previous_factor = restir_pairwise_m_factor(
        previous_at_previous, previous_at_current);
    const float current_factor = restir_pairwise_m_factor(
        current_at_previous, current_at_current);
    const float effective_factor =
        previous_factor < current_factor ? previous_factor : current_factor;
    const float previous_adjusted_M =
        previous_effective_M * effective_factor;

    reset_reservoir(output);
    constexpr std::uint32_t kTemporalPairwiseDomain = 0x54505753u;
    RNG rng(restir_random_seed(seed, pixel, iteration,
                              kTemporalPairwiseDomain));
    bool selected_history = false;
    const float previous_mass = previous_at_current *
                                previous.unbiased_contribution_weight * w0;
    if (previous_mass > 0.0f && previous_at_current > 0.0f) {
        const ReservoirOperationResult streamed = stream_di_weight(
            output, previous.sample, previous_at_current, previous_mass,
            previous_count, previous_adjusted_M,
            static_cast<float>(rng.next()));
        if (!streamed.accepted()) {
            return RestirDIStatus::ReservoirFailure;
        }
        selected_history = streamed.changed_selection();
        stats.accepted = 1u;
        stats.rejection = RestirTemporalRejection::Accepted;
    } else {
        const ReservoirOperationResult represented =
            represent_di_candidates(output, previous_count,
                                    previous_adjusted_M);
        if (represented.rejection != ReservoirRejectReason::ZeroTarget &&
            !represented.accepted()) {
            return RestirDIStatus::ReservoirFailure;
        }
        stats.rejection =
            RestirTemporalRejection::DestinationOutsideSupport;
    }

    const float canonical_weight = 1.0f - w1;
    const float current_mass = current_at_current *
                               current_reservoir.unbiased_contribution_weight *
                               canonical_weight;
    if (current_mass > 0.0f && current_at_current > 0.0f) {
        const ReservoirOperationResult streamed = stream_di_weight(
            output, current_reservoir.sample, current_at_current,
            current_mass, current_count, current_effective_M,
            static_cast<float>(rng.next()));
        if (!streamed.accepted()) {
            return RestirDIStatus::ReservoirFailure;
        }
        if (streamed.changed_selection()) {
            selected_history = false;
        }
    } else {
        const ReservoirOperationResult represented =
            represent_di_candidates(output, current_count,
                                    current_effective_M);
        if (represented.rejection != ReservoirRejectReason::ZeroTarget &&
            !represented.accepted()) {
            return RestirDIStatus::ReservoirFailure;
        }
    }
    const ReservoirOperationResult finalized =
        finalize_di_reservoir(output, 1.0f);
    if (!finalized.accepted()) {
        return finalized.rejection == ReservoirRejectReason::EmptyReservoir ||
                       finalized.rejection ==
                           ReservoirRejectReason::NoSelectedSample
                   ? RestirDIStatus::ReservoirEmpty
                   : RestirDIStatus::ReservoirFailure;
    }
    output.age = selected_history ? previous.age + 1u : 0u;
    return RestirDIStatus::Success;
}

} // namespace restir

#endif

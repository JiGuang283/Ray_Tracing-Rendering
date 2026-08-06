#ifndef RESTIR_GI_TEMPORAL_CORE_H
#define RESTIR_GI_TEMPORAL_CORE_H

#include "restir_gi_spatial_core.h"
#include "restir_reprojection_core.h"

namespace restir {

struct RestirGITemporalStats {
    std::uint32_t candidates = 0u;
    std::uint32_t accepted = 0u;
    RestirTemporalRejection rejection =
        RestirTemporalRejection::NoHistory;
    RestirGIShiftFailure shift_failure = RestirGIShiftFailure::None;
};

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
    stats.candidates = 1u;
    if (previous_camera == nullptr || previous_surfaces == nullptr ||
        previous_reservoirs == nullptr) {
        return RestirGIStatus::Success;
    }
    RestirReprojection projection;
    stats.rejection = reproject_restir_surface(
        *previous_camera, current_surface, width, height, projection);
    if (stats.rejection != RestirTemporalRejection::Accepted) {
        return RestirGIStatus::Success;
    }
    current_surface.motion = projection.previous_pixel;
    const RestirSurface &previous_surface =
        previous_surfaces[projection.previous_pixel];
    stats.rejection = restir_temporal_compatibility(
        current_surface, previous_surface,
        projection.expected_previous_depth, normal_threshold,
        depth_threshold);
    if (stats.rejection != RestirTemporalRejection::Accepted) {
        return RestirGIStatus::Success;
    }
    const RestirGIReservoir &previous =
        previous_reservoirs[projection.previous_pixel];
    if (!reservoir_is_usable(previous)) {
        stats.rejection = RestirTemporalRejection::InvalidReservoir;
        return RestirGIStatus::Success;
    }
    if (previous.age >= max_history_length) {
        stats.rejection = RestirTemporalRejection::AgeLimit;
        return RestirGIStatus::Success;
    }
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

} // namespace restir

#endif

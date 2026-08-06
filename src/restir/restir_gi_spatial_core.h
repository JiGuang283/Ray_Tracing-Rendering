#ifndef RESTIR_GI_SPATIAL_CORE_H
#define RESTIR_GI_SPATIAL_CORE_H

#include "restir_gi_core.h"
#include "restir_spatial_core.h"

namespace restir {

struct RestirGISpatialStats {
    std::uint32_t candidates = 0u;
    std::uint32_t accepted = 0u;
    std::uint32_t rejected = 0u;
    std::uint32_t compatibility[
        static_cast<std::uint32_t>(RestirSpatialCompatibility::Count)]{};
    std::uint32_t shift_failures[kRestirShiftFailureBuckets]{};
};

RT_HOST_DEVICE RT_FORCE_INLINE RestirGIStatus combine_basic_gi_source(
    const RestirDIPixelContext &destination,
    const RestirGIReservoir &source, std::uint32_t max_candidates,
    float random, RestirGIReservoir &output, bool &accepted,
    bool *changed_selection = nullptr,
    RestirGIShiftFailure *shift_failure = nullptr) noexcept {
    accepted = false;
    if (changed_selection != nullptr) {
        *changed_selection = false;
    }
    if (shift_failure != nullptr) {
        *shift_failure = RestirGIShiftFailure::None;
    }
    std::uint32_t represented_count = 0u;
    float effective_count = 0.0f;
    float mass_fraction = 0.0f;
    specialized_reservoir::capped_source_mass(
        source, output.M, max_candidates, represented_count,
        effective_count, mass_fraction);
    if (represented_count == 0u) {
        return RestirGIStatus::Success;
    }
    if (!reservoir_is_usable(source)) {
        const ReservoirOperationResult represented =
            represent_gi_candidates(output, represented_count,
                                    effective_count);
        return represented.accepted() ||
                       represented.rejection ==
                           ReservoirRejectReason::ZeroTarget
                   ? RestirGIStatus::Success
                   : RestirGIStatus::ReservoirFailure;
    }

    RestirGIReconnectResult reconnect;
    const RestirGIStatus reconnect_status =
        evaluate_diffuse_reconnection(destination, source.sample,
                                      reconnect);
    if (shift_failure != nullptr) {
        *shift_failure = reconnect.failure;
    }
    if (reconnect_status != RestirGIStatus::Success) {
        return reconnect_status;
    }
    if (!(reconnect.target > 0.0f)) {
        const ReservoirOperationResult represented =
            represent_gi_candidates(output, represented_count,
                                    effective_count);
        return represented.accepted() ||
                       represented.rejection ==
                           ReservoirRejectReason::ZeroTarget
                   ? RestirGIStatus::Success
                   : RestirGIStatus::ReservoirFailure;
    }
    const float stream_normalization =
        source.unbiased_contribution_weight * source.effective_M *
        mass_fraction;
    const ReservoirOperationResult combined = stream_gi_weight(
        output, source.sample, reconnect.target,
        reconnect.target * stream_normalization, represented_count,
        effective_count, random);
    if (!combined.accepted()) {
        return RestirGIStatus::ReservoirFailure;
    }
    accepted = true;
    if (changed_selection != nullptr) {
        *changed_selection = combined.changed_selection();
    }
    return RestirGIStatus::Success;
}

RT_HOST_DEVICE RT_FORCE_INLINE RestirGIStatus spatial_resample_gi_basic(
    const CompiledSceneView &scene, const RestirSurface *surfaces,
    const RestirGIReservoir *source_reservoirs,
    std::uint32_t width, std::uint32_t height, std::uint32_t pixel,
    std::uint32_t iteration, std::uint32_t pass_index,
    std::uint32_t seed, std::uint32_t neighbor_count,
    std::uint32_t max_candidates, float normal_threshold,
    float depth_threshold, RestirGIReservoir &output,
    RestirGISpatialStats &stats) noexcept {
    reset_reservoir(output);
    stats = {};
    if (surfaces == nullptr || source_reservoirs == nullptr || width < 2u ||
        height < 2u ||
        pixel >= static_cast<std::uint64_t>(width) * height ||
        neighbor_count > kMaxRestirSpatialNeighbors) {
        return RestirGIStatus::ReservoirFailure;
    }
    RestirDIPixelContext destination;
    const RestirGIStatus context_status = reconstruct_restir_gi_context(
        scene, surfaces[pixel], width, height, pixel, iteration, seed,
        destination);
    if (context_status != RestirGIStatus::Success) {
        return context_status;
    }
    constexpr std::uint32_t kSpatialDomain = 0x47495350u;
    RNG rng(restir_random_seed(seed, pixel, iteration,
                              kSpatialDomain ^ pass_index));
    auto combine = [&](const RestirGIReservoir &source) {
        ++stats.candidates;
        bool accepted = false;
        bool selected = false;
        RestirGIShiftFailure failure = RestirGIShiftFailure::None;
        const RestirGIStatus status = combine_basic_gi_source(
            destination, source, max_candidates,
            static_cast<float>(rng.next()), output, accepted, &selected,
            &failure);
        if (failure != RestirGIShiftFailure::None) {
            const std::uint32_t index = static_cast<std::uint32_t>(failure);
            if (index < kRestirShiftFailureBuckets) {
                ++stats.shift_failures[index];
            }
        }
        stats.accepted += accepted ? 1u : 0u;
        if (selected) {
            output.age = source.age;
        }
        return status;
    };

    RestirGIStatus status = combine(source_reservoirs[pixel]);
    if (status != RestirGIStatus::Success) {
        ++stats.rejected;
        return status;
    }
    for (std::uint32_t ordinal = 0u; ordinal < neighbor_count; ++ordinal) {
        if (max_candidates != 0u && output.M >= max_candidates) {
            break;
        }
        const RestirSpatialNeighbor neighbor = restir_spatial_neighbor(
            pixel, ordinal, width, height, iteration, pass_index, seed);
        if (neighbor.valid == 0u) {
            continue;
        }
        const RestirSpatialCompatibility compatibility =
            restir_spatial_compatibility(
                surfaces[pixel], surfaces[neighbor.pixel],
                normal_threshold, depth_threshold);
        ++stats.compatibility[static_cast<std::uint32_t>(compatibility)];
        if (compatibility != RestirSpatialCompatibility::Compatible) {
            ++stats.rejected;
            continue;
        }
        status = combine(source_reservoirs[neighbor.pixel]);
        if (status != RestirGIStatus::Success) {
            ++stats.rejected;
            return status;
        }
    }
    const ReservoirOperationResult finalized = finalize_reservoir(output);
    if (!finalized.accepted()) {
        return finalized.rejection == ReservoirRejectReason::EmptyReservoir ||
                       finalized.rejection ==
                           ReservoirRejectReason::NoSelectedSample
                   ? RestirGIStatus::ReservoirEmpty
                   : RestirGIStatus::ReservoirFailure;
    }
    return RestirGIStatus::Success;
}

} // namespace restir

#endif

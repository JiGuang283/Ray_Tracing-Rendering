#ifndef RESTIR_GI_SPATIAL_PAIRWISE_CORE_H
#define RESTIR_GI_SPATIAL_PAIRWISE_CORE_H

#include "restir_gi_pairwise_core.h"
#include "restir_gi_spatial_core.h"

namespace restir {

RT_HOST_DEVICE RT_FORCE_INLINE void record_gi_shift_failure(
    RestirGIShiftFailure failure, RestirGISpatialStats &stats) noexcept {
    const std::uint32_t index = static_cast<std::uint32_t>(failure);
    if (failure != RestirGIShiftFailure::None &&
        index < kRestirShiftFailureBuckets) {
        ++stats.shift_failures[index];
    }
}

RT_HOST_DEVICE RT_FORCE_INLINE RestirGIStatus evaluate_spatial_gi_target(
    const CompiledSceneView &scene, const RestirDIPixelContext &context,
    const RestirSurface &stored, const RestirGISample &sample,
    const PackedTransportSettings &transport, float &target,
    RestirGISpatialStats &stats) noexcept {
    RestirGIShiftFailure failure = RestirGIShiftFailure::None;
    std::uint32_t shadow_rays = 0u;
    std::uint32_t traversal_steps = 0u;
    const RestirGIStatus status = evaluate_gi_pairwise_target(
        scene, context, stored, sample, transport, target, failure,
        shadow_rays, traversal_steps);
    if (sample.random_replay()) {
        ++stats.replay_evaluations;
        stats.replay_shadow_rays += shadow_rays;
        stats.replay_traversal_steps += traversal_steps;
    }
    record_gi_shift_failure(failure, stats);
    return status;
}

RT_HOST_DEVICE RT_FORCE_INLINE RestirGIStatus spatial_resample_gi_pairwise(
    const CompiledSceneView &scene, const RestirSurface *surfaces,
    const RestirGIReservoir *source_reservoirs,
    std::uint32_t width, std::uint32_t height, std::uint32_t pixel,
    std::uint32_t iteration, std::uint32_t pass_index,
    std::uint32_t seed, std::uint32_t neighbor_count,
    std::uint32_t max_candidates, float normal_threshold,
    float depth_threshold, const PackedTransportSettings &transport,
    RestirGIReservoir &output,
    RestirGISpatialStats &stats) noexcept {
    reset_reservoir(output);
    stats = {};
    if (surfaces == nullptr || source_reservoirs == nullptr || width < 2u ||
        height < 2u ||
        pixel >= static_cast<std::uint64_t>(width) * height ||
        neighbor_count > kMaxRestirSpatialNeighbors) {
        return RestirGIStatus::ReservoirFailure;
    }
    const RestirGIReservoir &canonical = source_reservoirs[pixel];
    if (!reservoir_is_usable(canonical)) {
        const RestirGIStatus fallback = spatial_resample_gi_basic(
            scene, surfaces, source_reservoirs, width, height, pixel,
            iteration, pass_index, seed, neighbor_count, max_candidates,
            normal_threshold, depth_threshold, transport, output, stats);
        ++stats.pairwise_fallbacks;
        return fallback;
    }

    RestirDIPixelContext center_context;
    RestirGIStatus status = reconstruct_restir_gi_context(
        scene, surfaces[pixel], width, height, pixel, iteration, seed,
        center_context);
    if (status != RestirGIStatus::Success) {
        return status;
    }

    std::uint32_t canonical_count = 0u;
    float canonical_effective = 0.0f;
    float ignored_fraction = 0.0f;
    specialized_reservoir::capped_source_mass(
        canonical, 0u, max_candidates, canonical_count,
        canonical_effective, ignored_fraction);
    std::uint32_t planned_M = canonical_count;
    std::uint32_t valid_neighbors = 0u;
    for (std::uint32_t ordinal = 0u; ordinal < neighbor_count; ++ordinal) {
        if (max_candidates != 0u && planned_M >= max_candidates) {
            break;
        }
        const RestirSpatialNeighbor neighbor = restir_spatial_neighbor(
            pixel, ordinal, width, height, iteration, pass_index, seed);
        if (neighbor.valid == 0u ||
            restir_spatial_compatibility(
                surfaces[pixel], surfaces[neighbor.pixel],
                normal_threshold, depth_threshold) !=
                RestirSpatialCompatibility::Compatible) {
            continue;
        }
        std::uint32_t represented_count = 0u;
        float effective_count = 0.0f;
        specialized_reservoir::capped_source_mass(
            source_reservoirs[neighbor.pixel], planned_M, max_candidates,
            represented_count, effective_count, ignored_fraction);
        if (represented_count == 0u) {
            continue;
        }
        planned_M += represented_count;
        ++valid_neighbors;
    }

    constexpr std::uint32_t kSpatialPairwiseDomain = 0x47495057u;
    RNG rng(restir_random_seed(
        seed, pixel, iteration, kSpatialPairwiseDomain ^ pass_index));
    float canonical_weight = 0.0f;
    planned_M = canonical_count;
    for (std::uint32_t ordinal = 0u; ordinal < neighbor_count; ++ordinal) {
        if (max_candidates != 0u && planned_M >= max_candidates) {
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
                surfaces[pixel], surfaces[neighbor.pixel],
                normal_threshold, depth_threshold);
        ++stats.compatibility[static_cast<std::uint32_t>(compatibility)];
        if (compatibility != RestirSpatialCompatibility::Compatible) {
            ++stats.rejected;
            continue;
        }

        const RestirGIReservoir &neighbor_reservoir =
            source_reservoirs[neighbor.pixel];
        std::uint32_t represented_count = 0u;
        float effective_count = 0.0f;
        specialized_reservoir::capped_source_mass(
            neighbor_reservoir, planned_M, max_candidates,
            represented_count, effective_count, ignored_fraction);
        if (represented_count == 0u) {
            ++stats.rejected;
            continue;
        }
        planned_M += represented_count;

        RestirDIPixelContext neighbor_context;
        status = reconstruct_restir_gi_context(
            scene, surfaces[neighbor.pixel], width, height,
            neighbor.pixel, iteration, seed, neighbor_context);
        if (status != RestirGIStatus::Success) {
            ++stats.rejected;
            return status;
        }

        float neighbor_at_center = 0.0f;
        float neighbor_at_neighbor = 0.0f;
        if (reservoir_is_usable(neighbor_reservoir)) {
            status = evaluate_spatial_gi_target(
                scene, center_context, surfaces[pixel],
                neighbor_reservoir.sample, transport, neighbor_at_center,
                stats);
            if (status == RestirGIStatus::Success) {
                status = evaluate_spatial_gi_target(
                    scene, neighbor_context, surfaces[neighbor.pixel],
                    neighbor_reservoir.sample, transport,
                    neighbor_at_neighbor, stats);
            }
            if (status != RestirGIStatus::Success) {
                ++stats.rejected;
                return status;
            }
        }
        float canonical_at_neighbor = 0.0f;
        float canonical_at_center = 0.0f;
        status = evaluate_spatial_gi_target(
            scene, neighbor_context, surfaces[neighbor.pixel],
            canonical.sample, transport, canonical_at_neighbor, stats);
        if (status == RestirGIStatus::Success) {
            status = evaluate_spatial_gi_target(
                scene, center_context, surfaces[pixel], canonical.sample,
                transport, canonical_at_center, stats);
        }
        if (status != RestirGIStatus::Success) {
            ++stats.rejected;
            return status;
        }

        const float neighbor_estimator_M =
            effective_count * static_cast<float>(neighbor_count);
        const float w_neighbor = restir_pairwise_mis_weight(
            neighbor_at_neighbor, neighbor_at_center,
            neighbor_estimator_M, canonical_effective);
        const float w_canonical = restir_pairwise_mis_weight(
            canonical_at_neighbor, canonical_at_center,
            neighbor_estimator_M, canonical_effective);
        const float neighbor_factor = restir_pairwise_m_factor(
            neighbor_at_neighbor, neighbor_at_center);
        const float canonical_factor = restir_pairwise_m_factor(
            canonical_at_neighbor, canonical_at_center);
        const float adjusted_effective = effective_count *
            (neighbor_factor < canonical_factor ? neighbor_factor
                                                : canonical_factor);
        canonical_weight += 1.0f - w_canonical;
        const float mass = neighbor_at_center *
            neighbor_reservoir.unbiased_contribution_weight * w_neighbor;
        bool changed_selection = false;
        status = stream_pairwise_gi_mass(
            output, neighbor_reservoir.sample, neighbor_at_center, mass,
            represented_count, adjusted_effective,
            static_cast<float>(rng.next()),
            reservoir_is_usable(neighbor_reservoir), changed_selection);
        if (status != RestirGIStatus::Success) {
            ++stats.rejected;
            return status;
        }
        if (changed_selection) {
            output.age = neighbor_reservoir.age;
        }
        ++stats.accepted;
    }

    if (valid_neighbors == 0u) {
        canonical_weight = 1.0f;
    }
    ++stats.candidates;
    float canonical_target = 0.0f;
    status = evaluate_spatial_gi_target(
        scene, center_context, surfaces[pixel], canonical.sample,
        transport, canonical_target, stats);
    if (status != RestirGIStatus::Success) {
        ++stats.rejected;
        return status;
    }
    const float canonical_mass = canonical_target *
        canonical.unbiased_contribution_weight * canonical_weight;
    bool changed_selection = false;
    status = stream_pairwise_gi_mass(
        output, canonical.sample, canonical_target, canonical_mass,
        canonical_count, canonical_effective,
        static_cast<float>(rng.next()), true, changed_selection);
    if (status != RestirGIStatus::Success) {
        ++stats.rejected;
        return status;
    }
    if (changed_selection) {
        output.age = canonical.age;
    }
    ++stats.accepted;

    const float normalization =
        static_cast<float>(valid_neighbors > 0u ? valid_neighbors : 1u);
    const ReservoirOperationResult finalized =
        finalize_gi_reservoir(output, normalization);
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

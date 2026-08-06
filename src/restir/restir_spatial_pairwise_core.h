#ifndef RESTIR_SPATIAL_PAIRWISE_CORE_H
#define RESTIR_SPATIAL_PAIRWISE_CORE_H

#include "restir_spatial_core.h"

namespace restir {

RT_HOST_DEVICE RT_FORCE_INLINE float restir_pairwise_m_factor(
    float q0, float q1) noexcept {
    if (!(q0 > 0.0f)) {
        return 1.0f;
    }
    float ratio = q1 / q0;
    ratio = ratio < 0.0f ? 0.0f : (ratio > 1.0f ? 1.0f : ratio);
    const float squared = ratio * ratio;
    const float fourth = squared * squared;
    return fourth * fourth;
}

RT_HOST_DEVICE RT_FORCE_INLINE float restir_pairwise_mis_weight(
    float q0, float q1, float M0, float M1) noexcept {
    const float numerator = M0 * q0;
    const float denominator = numerator + M1 * q1;
    if (!(denominator > 0.0f) || !detail::finite(denominator) ||
        !detail::finite(numerator)) {
        return 0.0f;
    }
    const float weight = numerator / denominator;
    return weight < 0.0f ? 0.0f : (weight > 1.0f ? 1.0f : weight);
}

RT_HOST_DEVICE RT_FORCE_INLINE RestirDIStatus stream_pairwise_di_mass(
    RestirDIReservoir &output, const RestirLightSample &sample,
    float destination_target, float weight,
    std::uint32_t represented_count, float effective_count, float random,
    bool sample_valid) noexcept {
    if (!sample_valid || !(destination_target > 0.0f) || !(weight > 0.0f)) {
        const ReservoirOperationResult represented =
            represent_di_candidates(output, represented_count,
                                    effective_count);
        return represented.accepted() ||
                       represented.rejection ==
                           ReservoirRejectReason::ZeroTarget
                   ? RestirDIStatus::Success
                   : RestirDIStatus::ReservoirFailure;
    }
    const ReservoirOperationResult streamed = stream_di_weight(
        output, sample, destination_target, weight, represented_count,
        effective_count, random);
    return streamed.accepted() ? RestirDIStatus::Success
                               : RestirDIStatus::ReservoirFailure;
}

RT_HOST_DEVICE RT_FORCE_INLINE RestirDIStatus spatial_resample_di_pairwise(
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
    const RestirDIReservoir &canonical = source_reservoirs[pixel];
    if (!reservoir_is_usable(canonical)) {
        const RestirDIStatus fallback = spatial_resample_di_basic(
            scene, surfaces, source_reservoirs, width, height, pixel,
            iteration, pass_index, seed, neighbor_count, max_candidates,
            normal_threshold, depth_threshold, output, stats);
        ++stats.pairwise_fallbacks;
        return fallback;
    }

    RestirDIPixelContext center_context;
    const RestirDIStatus center_status = reconstruct_restir_di_context(
        scene, surfaces[pixel], width, height, pixel, iteration, seed,
        center_context);
    if (center_status != RestirDIStatus::Success) {
        return center_status;
    }

    std::uint32_t canonical_count = 0u;
    float canonical_effective_M = 0.0f;
    float canonical_fraction = 0.0f;
    capped_di_source_mass(canonical, 0u, max_candidates, canonical_count,
                          canonical_effective_M, canonical_fraction);
    std::uint32_t planned_M = canonical_count;
    std::uint32_t valid_neighbors = 0u;
    for (std::uint32_t ordinal = 0u; ordinal < neighbor_count; ++ordinal) {
        if (max_candidates != 0u && planned_M >= max_candidates) {
            break;
        }
        const RestirSpatialNeighbor neighbor = restir_spatial_neighbor(
            pixel, ordinal, width, height, iteration, pass_index, seed);
        if (neighbor.valid == 0u) {
            continue;
        }
        const RestirSpatialCompatibility compatibility =
            restir_spatial_compatibility(
                surfaces[pixel], surfaces[neighbor.pixel], normal_threshold,
                depth_threshold);
        if (compatibility != RestirSpatialCompatibility::Compatible) {
            continue;
        }
        std::uint32_t represented_count = 0u;
        float effective_count = 0.0f;
        float fraction = 0.0f;
        capped_di_source_mass(source_reservoirs[neighbor.pixel], planned_M,
                              max_candidates, represented_count,
                              effective_count, fraction);
        if (represented_count == 0u) {
            continue;
        }
        planned_M += represented_count;
        ++valid_neighbors;
    }

    constexpr std::uint32_t kPairwiseDomain = 0x50574953u;
    RNG rng(restir_random_seed(seed, pixel, iteration,
                              kPairwiseDomain ^ pass_index));
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
                surfaces[pixel], surfaces[neighbor.pixel], normal_threshold,
                depth_threshold);
        ++stats.compatibility[static_cast<std::uint32_t>(compatibility)];
        if (compatibility != RestirSpatialCompatibility::Compatible) {
            ++stats.rejected;
            continue;
        }
        const RestirDIReservoir &neighbor_reservoir =
            source_reservoirs[neighbor.pixel];
        std::uint32_t represented_count = 0u;
        float effective_count = 0.0f;
        float fraction = 0.0f;
        capped_di_source_mass(neighbor_reservoir, planned_M,
                              max_candidates, represented_count,
                              effective_count, fraction);
        if (represented_count == 0u) {
            ++stats.rejected;
            continue;
        }
        planned_M += represented_count;

        RestirDIPixelContext neighbor_context;
        const RestirDIStatus neighbor_context_status =
            reconstruct_restir_di_context(
                scene, surfaces[neighbor.pixel], width, height,
                neighbor.pixel, iteration, seed, neighbor_context);
        if (neighbor_context_status != RestirDIStatus::Success) {
            ++stats.rejected;
            return neighbor_context_status;
        }

        float neighbor_at_center = 0.0f;
        float neighbor_at_neighbor = 0.0f;
        if (reservoir_is_usable(neighbor_reservoir)) {
            RestirDIStatus target_status = evaluate_restir_sample_target(
                scene, center_context, neighbor_reservoir.sample,
                neighbor_at_center);
            if (target_status != RestirDIStatus::Success) {
                ++stats.rejected;
                return target_status;
            }
            target_status = evaluate_restir_sample_target(
                scene, neighbor_context, neighbor_reservoir.sample,
                neighbor_at_neighbor);
            if (target_status != RestirDIStatus::Success) {
                ++stats.rejected;
                return target_status;
            }
        }
        float canonical_at_neighbor = 0.0f;
        float canonical_at_center = 0.0f;
        RestirDIStatus target_status = evaluate_restir_sample_target(
            scene, neighbor_context, canonical.sample,
            canonical_at_neighbor);
        if (target_status != RestirDIStatus::Success) {
            ++stats.rejected;
            return target_status;
        }
        target_status = evaluate_restir_sample_target(
            scene, center_context, canonical.sample, canonical_at_center);
        if (target_status != RestirDIStatus::Success) {
            ++stats.rejected;
            return target_status;
        }

        const float neighbor_M = effective_count;
        const float neighbor_estimator_M =
            neighbor_M * static_cast<float>(neighbor_count);
        const float w0 = restir_pairwise_mis_weight(
            neighbor_at_neighbor, neighbor_at_center,
            neighbor_estimator_M, canonical_effective_M);
        const float w1 = restir_pairwise_mis_weight(
            canonical_at_neighbor, canonical_at_center,
            neighbor_estimator_M, canonical_effective_M);
        const float neighbor_factor = restir_pairwise_m_factor(
            neighbor_at_neighbor, neighbor_at_center);
        const float canonical_factor = restir_pairwise_m_factor(
            canonical_at_neighbor, canonical_at_center);
        const float effective_factor =
            neighbor_factor < canonical_factor ? neighbor_factor
                                               : canonical_factor;
        const float adjusted_effective_M = neighbor_M * effective_factor;
        canonical_weight += 1.0f - w1;
        const float weight =
            neighbor_at_center * neighbor_reservoir.unbiased_contribution_weight *
            w0;
        const RestirDIStatus stream_status = stream_pairwise_di_mass(
            output, neighbor_reservoir.sample, neighbor_at_center, weight,
            represented_count, adjusted_effective_M,
            static_cast<float>(rng.next()),
            reservoir_is_usable(neighbor_reservoir));
        if (stream_status != RestirDIStatus::Success) {
            ++stats.rejected;
            return stream_status;
        }
        ++stats.accepted;
    }

    if (valid_neighbors == 0u) {
        canonical_weight = 1.0f;
    }
    ++stats.candidates;
    float canonical_target = 0.0f;
    const RestirDIStatus canonical_target_status =
        evaluate_restir_sample_target(scene, center_context,
                                      canonical.sample, canonical_target);
    if (canonical_target_status != RestirDIStatus::Success) {
        ++stats.rejected;
        return canonical_target_status;
    }
    const float canonical_mass = canonical_target *
                                 canonical.unbiased_contribution_weight *
                                 canonical_weight;
    const RestirDIStatus canonical_stream_status = stream_pairwise_di_mass(
        output, canonical.sample, canonical_target, canonical_mass,
        canonical_count, canonical_effective_M,
        static_cast<float>(rng.next()), true);
    if (canonical_stream_status != RestirDIStatus::Success) {
        ++stats.rejected;
        return canonical_stream_status;
    }
    ++stats.accepted;

    const float normalization =
        static_cast<float>(valid_neighbors > 0u ? valid_neighbors : 1u);
    const ReservoirOperationResult finalized =
        finalize_di_reservoir(output, normalization);
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

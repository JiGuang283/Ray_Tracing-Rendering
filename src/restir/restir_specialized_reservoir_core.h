#ifndef RESTIR_SPECIALIZED_RESERVOIR_CORE_H
#define RESTIR_SPECIALIZED_RESERVOIR_CORE_H

#include "reservoir_core.h"

#include <limits>

namespace restir {
namespace specialized_reservoir {

template <typename Reservoir>
RT_HOST_DEVICE RT_FORCE_INLINE bool
raw_state_valid(const Reservoir &reservoir) noexcept {
    if ((reservoir.flags & RESERVOIR_FLAG_FINALIZED) != 0u ||
        !detail::finite(reservoir.weight_sum) || reservoir.weight_sum < 0.0f ||
        !detail::finite(reservoir.effective_M) || reservoir.effective_M < 0.0f ||
        reservoir.unbiased_contribution_weight != 0.0f) {
        return false;
    }
    const bool has_sample =
        (reservoir.flags & RESERVOIR_FLAG_HAS_SAMPLE) != 0u;
    if (!has_sample) {
        return reservoir.weight_sum == 0.0f &&
               reservoir.selected_target == 0.0f;
    }
    return reservoir.weight_sum > 0.0f &&
           detail::finite(reservoir.selected_target) &&
           reservoir.selected_target > 0.0f;
}

template <typename Reservoir>
RT_HOST_DEVICE RT_FORCE_INLINE void reset(Reservoir &reservoir) noexcept {
    reservoir = Reservoir{};
}

template <typename Reservoir>
RT_HOST_DEVICE RT_FORCE_INLINE bool
has_sample(const Reservoir &reservoir) noexcept {
    return (reservoir.flags & RESERVOIR_FLAG_HAS_SAMPLE) != 0u;
}

template <typename Reservoir>
RT_HOST_DEVICE RT_FORCE_INLINE bool
is_finalized(const Reservoir &reservoir) noexcept {
    return (reservoir.flags & RESERVOIR_FLAG_FINALIZED) != 0u;
}

template <typename Reservoir>
RT_HOST_DEVICE RT_FORCE_INLINE bool
is_usable(const Reservoir &reservoir) noexcept {
    return has_sample(reservoir) && is_finalized(reservoir) &&
           reservoir.M > 0u && detail::finite(reservoir.effective_M) &&
           reservoir.effective_M > 0.0f &&
           detail::finite(reservoir.weight_sum) &&
           reservoir.weight_sum > 0.0f &&
           detail::finite(reservoir.selected_target) &&
           reservoir.selected_target > 0.0f &&
           detail::finite(reservoir.unbiased_contribution_weight) &&
           reservoir.unbiased_contribution_weight > 0.0f;
}

template <typename Reservoir, typename Sample>
RT_HOST_DEVICE RT_FORCE_INLINE ReservoirOperationResult stream_weight(
    Reservoir &reservoir, const Sample &sample, float selected_target,
    float weight, std::uint32_t represented_count, float effective_count,
    float random) noexcept {
    if (!raw_state_valid(reservoir)) {
        return detail::operation_result(
            is_finalized(reservoir)
                ? ReservoirRejectReason::ReservoirFinalized
                : ReservoirRejectReason::InvalidReservoirState);
    }
    if (represented_count == 0u || effective_count < 0.0f) {
        return detail::operation_result(ReservoirRejectReason::EmptyReservoir);
    }
    if (reservoir.M > std::numeric_limits<std::uint32_t>::max() -
                          represented_count) {
        return detail::operation_result(
            ReservoirRejectReason::CandidateCountOverflow);
    }
    if (!detail::finite(effective_count)) {
        return detail::operation_result(
            ReservoirRejectReason::NonFiniteUnbiasedContributionWeight);
    }
    if (!detail::finite(random)) {
        return detail::operation_result(ReservoirRejectReason::NonFiniteRandom);
    }
    if (random < 0.0f || random >= 1.0f) {
        return detail::operation_result(ReservoirRejectReason::RandomOutOfRange);
    }
    if (!detail::finite(selected_target)) {
        return detail::operation_result(
            ReservoirRejectReason::NonFiniteSelectedTarget);
    }
    if (!(selected_target > 0.0f)) {
        return detail::operation_result(
            ReservoirRejectReason::NonPositiveSelectedTarget);
    }
    if (!detail::finite(weight)) {
        return detail::operation_result(ReservoirRejectReason::NonFiniteWeight);
    }
    if (!(weight > 0.0f)) {
        return detail::operation_result(
            ReservoirRejectReason::NonPositiveWeight);
    }

    const float new_weight_sum = reservoir.weight_sum + weight;
    const float new_effective_M = reservoir.effective_M + effective_count;
    if (!detail::finite(new_weight_sum)) {
        return detail::operation_result(
            ReservoirRejectReason::NonFiniteWeightSum);
    }
    if (!detail::finite(new_effective_M)) {
        return detail::operation_result(
            ReservoirRejectReason::NonFiniteUnbiasedContributionWeight);
    }
    const bool select = !has_sample(reservoir) ||
                        random * new_weight_sum < weight;
    reservoir.weight_sum = new_weight_sum;
    reservoir.effective_M = new_effective_M;
    reservoir.M += represented_count;
    if (select) {
        reservoir.sample = sample;
        reservoir.selected_target = selected_target;
        reservoir.flags |= RESERVOIR_FLAG_HAS_SAMPLE;
    }
    return detail::operation_result(ReservoirRejectReason::None, select,
                                    represented_count);
}

template <typename Reservoir>
RT_HOST_DEVICE RT_FORCE_INLINE ReservoirOperationResult represent_candidates(
    Reservoir &reservoir, std::uint32_t represented_count,
    float effective_count) noexcept {
    if (!raw_state_valid(reservoir)) {
        return detail::operation_result(
            is_finalized(reservoir)
                ? ReservoirRejectReason::ReservoirFinalized
                : ReservoirRejectReason::InvalidReservoirState);
    }
    if (represented_count == 0u || effective_count < 0.0f) {
        return detail::operation_result(ReservoirRejectReason::EmptyReservoir);
    }
    if (reservoir.M > std::numeric_limits<std::uint32_t>::max() -
                          represented_count) {
        return detail::operation_result(
            ReservoirRejectReason::CandidateCountOverflow);
    }
    const float new_effective_M = reservoir.effective_M + effective_count;
    if (!detail::finite(effective_count) ||
        !detail::finite(new_effective_M)) {
        return detail::operation_result(
            ReservoirRejectReason::NonFiniteUnbiasedContributionWeight);
    }
    reservoir.M += represented_count;
    reservoir.effective_M = new_effective_M;
    return detail::operation_result(ReservoirRejectReason::ZeroTarget, false,
                                    represented_count);
}

template <typename Reservoir>
RT_HOST_DEVICE RT_FORCE_INLINE void capped_source_mass(
    const Reservoir &source, std::uint32_t current_M,
    std::uint32_t max_candidates, std::uint32_t &represented_count,
    float &effective_count, float &mass_fraction) noexcept {
    represented_count = source.M;
    effective_count = source.effective_M;
    mass_fraction = 1.0f;
    if (max_candidates == 0u || current_M >= max_candidates ||
        source.M == 0u) {
        if (max_candidates != 0u && current_M >= max_candidates) {
            represented_count = 0u;
            effective_count = 0.0f;
            mass_fraction = 0.0f;
        }
        return;
    }
    const std::uint32_t remaining = max_candidates - current_M;
    if (represented_count > remaining) {
        represented_count = remaining;
        mass_fraction = static_cast<float>(remaining) /
                        static_cast<float>(source.M);
        effective_count *= mass_fraction;
    }
}

template <typename Reservoir, typename Candidate>
RT_HOST_DEVICE RT_FORCE_INLINE ReservoirOperationResult stream_candidate(
    Reservoir &reservoir, const Candidate &candidate, float random) noexcept {
    if ((candidate.flags & CANDIDATE_FLAG_VALID_SAMPLE) == 0u) {
        return detail::operation_result(ReservoirRejectReason::InvalidSample);
    }
    if (is_finalized(reservoir)) {
        return detail::operation_result(
            ReservoirRejectReason::ReservoirFinalized);
    }
    if (!raw_state_valid(reservoir)) {
        return detail::operation_result(
            ReservoirRejectReason::InvalidReservoirState);
    }
    if (!detail::finite(candidate.target)) {
        return detail::operation_result(ReservoirRejectReason::NonFiniteTarget);
    }
    if (candidate.target < 0.0f) {
        return detail::operation_result(ReservoirRejectReason::NegativeTarget);
    }
    if (!detail::finite(candidate.proposal_pdf)) {
        return detail::operation_result(
            ReservoirRejectReason::NonFiniteProposalPdf);
    }
    if (!(candidate.proposal_pdf > 0.0f)) {
        const bool nonzero =
            (candidate.flags & CANDIDATE_FLAG_NONZERO_CONTRIBUTION) != 0u;
        return detail::operation_result(
            candidate.proposal_pdf == 0.0f && nonzero
                ? ReservoirRejectReason::ZeroProposalWithContribution
                : ReservoirRejectReason::NonPositiveProposalPdf);
    }
    if (candidate.target == 0.0f) {
        if ((candidate.flags & CANDIDATE_FLAG_NONZERO_CONTRIBUTION) != 0u) {
            return detail::operation_result(
                ReservoirRejectReason::ZeroTargetWithContribution);
        }
        return represent_candidates(reservoir, 1u, 1.0f);
    }
    return stream_weight(reservoir, candidate.sample, candidate.target,
                         candidate.target / candidate.proposal_pdf, 1u,
                         1.0f, random);
}

template <typename Reservoir>
RT_HOST_DEVICE RT_FORCE_INLINE ReservoirOperationResult finalize(
    Reservoir &reservoir, float normalization_denominator) noexcept {
    if (is_finalized(reservoir)) {
        return is_usable(reservoir)
                   ? detail::operation_result(ReservoirRejectReason::None)
                   : detail::operation_result(
                         ReservoirRejectReason::InvalidReservoirState);
    }
    if (reservoir.M == 0u || !(reservoir.effective_M > 0.0f) ||
        !(normalization_denominator > 0.0f)) {
        return detail::operation_result(ReservoirRejectReason::EmptyReservoir);
    }
    if (!has_sample(reservoir)) {
        return detail::operation_result(
            ReservoirRejectReason::NoSelectedSample);
    }
    if (!detail::finite(reservoir.weight_sum) ||
        !(reservoir.weight_sum > 0.0f)) {
        return detail::operation_result(
            ReservoirRejectReason::NonFiniteWeightSum);
    }
    if (!detail::finite(reservoir.selected_target)) {
        return detail::operation_result(
            ReservoirRejectReason::NonFiniteSelectedTarget);
    }
    if (!(reservoir.selected_target > 0.0f)) {
        return detail::operation_result(
            ReservoirRejectReason::NonPositiveSelectedTarget);
    }
    if (!detail::finite(normalization_denominator)) {
        return detail::operation_result(
            ReservoirRejectReason::NonFiniteUnbiasedContributionWeight);
    }
    const float contribution_weight =
        (reservoir.weight_sum / reservoir.selected_target) /
        normalization_denominator;
    if (!detail::finite(contribution_weight) ||
        !(contribution_weight > 0.0f)) {
        return detail::operation_result(
            ReservoirRejectReason::NonFiniteUnbiasedContributionWeight);
    }
    reservoir.unbiased_contribution_weight = contribution_weight;
    reservoir.flags |= RESERVOIR_FLAG_FINALIZED;
    return detail::operation_result(ReservoirRejectReason::None);
}

} // namespace specialized_reservoir
} // namespace restir

#endif

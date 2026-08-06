#ifndef RESTIR_DI_RESERVOIR_CORE_H
#define RESTIR_DI_RESERVOIR_CORE_H

#include "reservoir_core.h"
#include "restir_di_types.h"

#include <limits>

namespace restir {
namespace detail {

RT_HOST_DEVICE RT_FORCE_INLINE bool
di_raw_state_valid(const RestirDIReservoir &reservoir) noexcept {
    if ((reservoir.flags & RESERVOIR_FLAG_FINALIZED) != 0u ||
        !finite(reservoir.weight_sum) || reservoir.weight_sum < 0.0f ||
        !finite(reservoir.effective_M) || reservoir.effective_M < 0.0f ||
        reservoir.unbiased_contribution_weight != 0.0f) {
        return false;
    }
    const bool has_sample =
        (reservoir.flags & RESERVOIR_FLAG_HAS_SAMPLE) != 0u;
    if (!has_sample) {
        return reservoir.weight_sum == 0.0f &&
               reservoir.selected_target == 0.0f;
    }
    return reservoir.weight_sum > 0.0f && reservoir.effective_M > 0.0f &&
           finite(reservoir.selected_target) &&
           reservoir.selected_target > 0.0f;
}

} // namespace detail

RT_HOST_DEVICE RT_FORCE_INLINE void
reset_reservoir(RestirDIReservoir &reservoir) noexcept {
    reservoir = RestirDIReservoir{};
}

RT_HOST_DEVICE RT_FORCE_INLINE bool
reservoir_has_sample(const RestirDIReservoir &reservoir) noexcept {
    return (reservoir.flags & RESERVOIR_FLAG_HAS_SAMPLE) != 0u;
}

RT_HOST_DEVICE RT_FORCE_INLINE bool
reservoir_is_finalized(const RestirDIReservoir &reservoir) noexcept {
    return (reservoir.flags & RESERVOIR_FLAG_FINALIZED) != 0u;
}

RT_HOST_DEVICE RT_FORCE_INLINE bool
reservoir_is_usable(const RestirDIReservoir &reservoir) noexcept {
    return reservoir_has_sample(reservoir) &&
           reservoir_is_finalized(reservoir) && reservoir.M > 0u &&
           detail::finite(reservoir.effective_M) &&
           reservoir.effective_M > 0.0f &&
           detail::finite(reservoir.weight_sum) &&
           reservoir.weight_sum > 0.0f &&
           detail::finite(reservoir.selected_target) &&
           reservoir.selected_target > 0.0f &&
           detail::finite(reservoir.unbiased_contribution_weight) &&
           reservoir.unbiased_contribution_weight > 0.0f;
}

RT_HOST_DEVICE RT_FORCE_INLINE ReservoirOperationResult stream_di_weight(
    RestirDIReservoir &reservoir, const RestirLightSample &sample,
    float selected_target, float weight, std::uint32_t represented_count,
    float effective_count, float random) noexcept {
    if (!detail::di_raw_state_valid(reservoir)) {
        return detail::operation_result(
            reservoir_is_finalized(reservoir)
                ? ReservoirRejectReason::ReservoirFinalized
                : ReservoirRejectReason::InvalidReservoirState);
    }
    if (represented_count == 0u || !(effective_count > 0.0f)) {
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

    const bool select = !reservoir_has_sample(reservoir) ||
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

RT_HOST_DEVICE RT_FORCE_INLINE ReservoirOperationResult
represent_di_candidates(RestirDIReservoir &reservoir,
                        std::uint32_t represented_count,
                        float effective_count) noexcept {
    if (!detail::di_raw_state_valid(reservoir)) {
        return detail::operation_result(
            reservoir_is_finalized(reservoir)
                ? ReservoirRejectReason::ReservoirFinalized
                : ReservoirRejectReason::InvalidReservoirState);
    }
    if (represented_count == 0u || !(effective_count > 0.0f)) {
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

RT_HOST_DEVICE RT_FORCE_INLINE ReservoirOperationResult stream_candidate(
    RestirDIReservoir &reservoir, const RestirDICandidate &candidate,
    float random) noexcept {
    if ((candidate.flags & CANDIDATE_FLAG_VALID_SAMPLE) == 0u) {
        return detail::operation_result(ReservoirRejectReason::InvalidSample);
    }
    if (reservoir_is_finalized(reservoir)) {
        return detail::operation_result(
            ReservoirRejectReason::ReservoirFinalized);
    }
    if (!detail::di_raw_state_valid(reservoir)) {
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
        if (reservoir.M == std::numeric_limits<std::uint32_t>::max()) {
            return detail::operation_result(
                ReservoirRejectReason::CandidateCountOverflow);
        }
        const float effective_M = reservoir.effective_M + 1.0f;
        if (!detail::finite(effective_M)) {
            return detail::operation_result(
                ReservoirRejectReason::NonFiniteUnbiasedContributionWeight);
        }
        ++reservoir.M;
        reservoir.effective_M = effective_M;
        return detail::operation_result(ReservoirRejectReason::ZeroTarget,
                                        false, 1u);
    }
    const float weight = candidate.target / candidate.proposal_pdf;
    return stream_di_weight(reservoir, candidate.sample, candidate.target,
                            weight, 1u, 1.0f, random);
}

RT_HOST_DEVICE RT_FORCE_INLINE ReservoirOperationResult
finalize_reservoir(RestirDIReservoir &reservoir) noexcept {
    if (reservoir_is_finalized(reservoir)) {
        return reservoir_is_usable(reservoir)
                   ? detail::operation_result(ReservoirRejectReason::None)
                   : detail::operation_result(
                         ReservoirRejectReason::InvalidReservoirState);
    }
    if (reservoir.M == 0u || !(reservoir.effective_M > 0.0f)) {
        return detail::operation_result(ReservoirRejectReason::EmptyReservoir);
    }
    if (!reservoir_has_sample(reservoir)) {
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
    const float contribution_weight =
        (reservoir.weight_sum / reservoir.selected_target) /
        reservoir.effective_M;
    if (!detail::finite(contribution_weight) ||
        !(contribution_weight > 0.0f)) {
        return detail::operation_result(
            ReservoirRejectReason::NonFiniteUnbiasedContributionWeight);
    }
    reservoir.unbiased_contribution_weight = contribution_weight;
    reservoir.flags |= RESERVOIR_FLAG_FINALIZED;
    return detail::operation_result(ReservoirRejectReason::None);
}

} // namespace restir

#endif

#ifndef RESTIR_RESERVOIR_CORE_H
#define RESTIR_RESERVOIR_CORE_H

#include "restir_types.h"

#include <limits>

namespace restir {
namespace detail {

template <typename Scalar>
RT_HOST_DEVICE RT_FORCE_INLINE bool finite(Scalar value) noexcept {
    const Scalar maximum = std::numeric_limits<Scalar>::max();
    return value == value && value <= maximum && value >= -maximum;
}

RT_HOST_DEVICE RT_FORCE_INLINE ReservoirOperationResult
operation_result(ReservoirRejectReason rejection, bool selected = false,
                 std::uint32_t represented_candidates = 0) noexcept {
    ReservoirOperationResult result;
    result.rejection = rejection;
    result.selected = selected ? 1u : 0u;
    result.represented_candidates = represented_candidates;
    return result;
}

template <typename Sample, typename Scalar>
RT_HOST_DEVICE RT_FORCE_INLINE bool
raw_state_valid(const Reservoir<Sample, Scalar> &reservoir) noexcept {
    if ((reservoir.flags & RESERVOIR_FLAG_FINALIZED) != 0u ||
        !finite(reservoir.weight_sum) || reservoir.weight_sum < Scalar(0) ||
        reservoir.unbiased_contribution_weight != Scalar(0)) {
        return false;
    }
    const bool has_sample =
        (reservoir.flags & RESERVOIR_FLAG_HAS_SAMPLE) != 0u;
    if (!has_sample) {
        return reservoir.weight_sum == Scalar(0) &&
               reservoir.selected_target == Scalar(0);
    }
    return reservoir.weight_sum > Scalar(0) &&
           finite(reservoir.selected_target) &&
           reservoir.selected_target > Scalar(0);
}

template <typename Sample, typename Scalar>
RT_HOST_DEVICE RT_FORCE_INLINE ReservoirOperationResult stream_weight(
    Reservoir<Sample, Scalar> &reservoir, const Sample &sample,
    Scalar selected_target, Scalar weight, std::uint32_t represented_count,
    Scalar random) noexcept {
    if (!raw_state_valid(reservoir)) {
        return operation_result(
            (reservoir.flags & RESERVOIR_FLAG_FINALIZED) != 0u
                ? ReservoirRejectReason::ReservoirFinalized
                : ReservoirRejectReason::InvalidReservoirState);
    }
    if (represented_count == 0u) {
        return operation_result(ReservoirRejectReason::EmptyReservoir);
    }
    if (reservoir.M >
        std::numeric_limits<std::uint32_t>::max() - represented_count) {
        return operation_result(
            ReservoirRejectReason::CandidateCountOverflow);
    }
    if (!finite(random)) {
        return operation_result(ReservoirRejectReason::NonFiniteRandom);
    }
    if (random < Scalar(0) || random >= Scalar(1)) {
        return operation_result(ReservoirRejectReason::RandomOutOfRange);
    }
    if (!finite(selected_target)) {
        return operation_result(
            ReservoirRejectReason::NonFiniteSelectedTarget);
    }
    if (selected_target <= Scalar(0)) {
        return operation_result(
            ReservoirRejectReason::NonPositiveSelectedTarget);
    }
    if (!finite(weight)) {
        return operation_result(ReservoirRejectReason::NonFiniteWeight);
    }
    if (weight <= Scalar(0)) {
        return operation_result(ReservoirRejectReason::NonPositiveWeight);
    }

    const Scalar new_weight_sum = reservoir.weight_sum + weight;
    if (!finite(new_weight_sum)) {
        return operation_result(ReservoirRejectReason::NonFiniteWeightSum);
    }

    const bool select =
        (reservoir.flags & RESERVOIR_FLAG_HAS_SAMPLE) == 0u ||
        random * new_weight_sum < weight;
    reservoir.weight_sum = new_weight_sum;
    reservoir.M += represented_count;
    if (select) {
        reservoir.sample = sample;
        reservoir.selected_target = selected_target;
        reservoir.flags |= RESERVOIR_FLAG_HAS_SAMPLE;
    }
    return operation_result(ReservoirRejectReason::None, select,
                            represented_count);
}

} // namespace detail

template <typename Sample, typename Scalar>
RT_HOST_DEVICE RT_FORCE_INLINE void
reset_reservoir(Reservoir<Sample, Scalar> &reservoir) noexcept {
    reservoir.sample = Sample{};
    reservoir.weight_sum = Scalar(0);
    reservoir.selected_target = Scalar(0);
    reservoir.unbiased_contribution_weight = Scalar(0);
    reservoir.M = 0;
    reservoir.flags = RESERVOIR_FLAG_NONE;
}

template <typename Sample, typename Scalar>
RT_HOST_DEVICE RT_FORCE_INLINE bool
reservoir_has_sample(const Reservoir<Sample, Scalar> &reservoir) noexcept {
    return (reservoir.flags & RESERVOIR_FLAG_HAS_SAMPLE) != 0u;
}

template <typename Sample, typename Scalar>
RT_HOST_DEVICE RT_FORCE_INLINE bool
reservoir_is_finalized(const Reservoir<Sample, Scalar> &reservoir) noexcept {
    return (reservoir.flags & RESERVOIR_FLAG_FINALIZED) != 0u;
}

template <typename Sample, typename Scalar>
RT_HOST_DEVICE RT_FORCE_INLINE bool
reservoir_is_usable(const Reservoir<Sample, Scalar> &reservoir) noexcept {
    return reservoir_has_sample(reservoir) &&
           reservoir_is_finalized(reservoir) && reservoir.M > 0u &&
           detail::finite(reservoir.weight_sum) &&
           reservoir.weight_sum > Scalar(0) &&
           detail::finite(reservoir.selected_target) &&
           reservoir.selected_target > Scalar(0) &&
           detail::finite(reservoir.unbiased_contribution_weight) &&
           reservoir.unbiased_contribution_weight > Scalar(0);
}

template <typename Sample, typename Scalar>
RT_HOST_DEVICE RT_FORCE_INLINE ReservoirOperationResult stream_candidate(
    Reservoir<Sample, Scalar> &reservoir,
    const ReservoirCandidate<Sample, Scalar> &candidate,
    Scalar random) noexcept {
    if ((candidate.flags & CANDIDATE_FLAG_VALID_SAMPLE) == 0u) {
        return detail::operation_result(ReservoirRejectReason::InvalidSample);
    }
    if ((reservoir.flags & RESERVOIR_FLAG_FINALIZED) != 0u) {
        return detail::operation_result(
            ReservoirRejectReason::ReservoirFinalized);
    }
    if (!detail::raw_state_valid(reservoir)) {
        return detail::operation_result(
            ReservoirRejectReason::InvalidReservoirState);
    }
    if (!detail::finite(candidate.target)) {
        return detail::operation_result(
            ReservoirRejectReason::NonFiniteTarget);
    }
    if (candidate.target < Scalar(0)) {
        return detail::operation_result(ReservoirRejectReason::NegativeTarget);
    }
    if (!detail::finite(candidate.proposal_pdf)) {
        return detail::operation_result(
            ReservoirRejectReason::NonFiniteProposalPdf);
    }
    if (candidate.proposal_pdf <= Scalar(0)) {
        const bool nonzero =
            (candidate.flags & CANDIDATE_FLAG_NONZERO_CONTRIBUTION) != 0u;
        return detail::operation_result(
            candidate.proposal_pdf == Scalar(0) && nonzero
                ? ReservoirRejectReason::ZeroProposalWithContribution
                : ReservoirRejectReason::NonPositiveProposalPdf);
    }

    if (candidate.target == Scalar(0)) {
        if ((candidate.flags & CANDIDATE_FLAG_NONZERO_CONTRIBUTION) != 0u) {
            return detail::operation_result(
                ReservoirRejectReason::ZeroTargetWithContribution);
        }
        if (reservoir.M == std::numeric_limits<std::uint32_t>::max()) {
            return detail::operation_result(
                ReservoirRejectReason::CandidateCountOverflow);
        }
        // A zero-weight proposal cannot be selected, but excluding the draw
        // from M would bias the RIS normalization.
        ++reservoir.M;
        return detail::operation_result(ReservoirRejectReason::ZeroTarget,
                                        false, 1u);
    }

    const Scalar weight = candidate.target / candidate.proposal_pdf;
    if (!detail::finite(weight)) {
        return detail::operation_result(ReservoirRejectReason::NonFiniteWeight);
    }
    if (weight <= Scalar(0)) {
        return detail::operation_result(
            ReservoirRejectReason::NonPositiveWeight);
    }
    return detail::stream_weight(reservoir, candidate.sample,
                                 candidate.target, weight, 1u, random);
}

template <typename Sample, typename Scalar>
RT_HOST_DEVICE RT_FORCE_INLINE ReservoirOperationResult
finalize_reservoir(Reservoir<Sample, Scalar> &reservoir) noexcept {
    if (reservoir_is_finalized(reservoir)) {
        return reservoir_is_usable(reservoir)
                   ? detail::operation_result(ReservoirRejectReason::None)
                   : detail::operation_result(
                         ReservoirRejectReason::InvalidReservoirState);
    }
    if (reservoir.M == 0u) {
        return detail::operation_result(ReservoirRejectReason::EmptyReservoir);
    }
    if (!reservoir_has_sample(reservoir)) {
        return detail::operation_result(
            ReservoirRejectReason::NoSelectedSample);
    }
    if (!detail::finite(reservoir.weight_sum) ||
        reservoir.weight_sum <= Scalar(0)) {
        return detail::operation_result(
            ReservoirRejectReason::NonFiniteWeightSum);
    }
    if (!detail::finite(reservoir.selected_target)) {
        return detail::operation_result(
            ReservoirRejectReason::NonFiniteSelectedTarget);
    }
    if (reservoir.selected_target <= Scalar(0)) {
        return detail::operation_result(
            ReservoirRejectReason::NonPositiveSelectedTarget);
    }

    const Scalar contribution_weight =
        (reservoir.weight_sum / reservoir.selected_target) /
        static_cast<Scalar>(reservoir.M);
    if (!detail::finite(contribution_weight) ||
        contribution_weight <= Scalar(0)) {
        return detail::operation_result(
            ReservoirRejectReason::NonFiniteUnbiasedContributionWeight);
    }
    reservoir.unbiased_contribution_weight = contribution_weight;
    reservoir.flags |= RESERVOIR_FLAG_FINALIZED;
    return detail::operation_result(ReservoirRejectReason::None);
}

} // namespace restir

#endif

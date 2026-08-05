#include "restir_validation.h"

namespace restir {

const char *reservoir_reject_reason_name(
    ReservoirRejectReason reason) noexcept {
    switch (reason) {
    case ReservoirRejectReason::None:
        return "none";
    case ReservoirRejectReason::InvalidSample:
        return "invalid_sample";
    case ReservoirRejectReason::ReservoirFinalized:
        return "reservoir_finalized";
    case ReservoirRejectReason::InvalidReservoirState:
        return "invalid_reservoir_state";
    case ReservoirRejectReason::NonFiniteRandom:
        return "non_finite_random";
    case ReservoirRejectReason::RandomOutOfRange:
        return "random_out_of_range";
    case ReservoirRejectReason::NonFiniteTarget:
        return "non_finite_target";
    case ReservoirRejectReason::NegativeTarget:
        return "negative_target";
    case ReservoirRejectReason::ZeroTarget:
        return "zero_target";
    case ReservoirRejectReason::ZeroTargetWithContribution:
        return "zero_target_with_contribution";
    case ReservoirRejectReason::NonFiniteProposalPdf:
        return "non_finite_proposal_pdf";
    case ReservoirRejectReason::NonPositiveProposalPdf:
        return "non_positive_proposal_pdf";
    case ReservoirRejectReason::ZeroProposalWithContribution:
        return "zero_proposal_with_contribution";
    case ReservoirRejectReason::NonFiniteWeight:
        return "non_finite_weight";
    case ReservoirRejectReason::NonPositiveWeight:
        return "non_positive_weight";
    case ReservoirRejectReason::NonFiniteWeightSum:
        return "non_finite_weight_sum";
    case ReservoirRejectReason::CandidateCountOverflow:
        return "candidate_count_overflow";
    case ReservoirRejectReason::EmptyReservoir:
        return "empty_reservoir";
    case ReservoirRejectReason::NoSelectedSample:
        return "no_selected_sample";
    case ReservoirRejectReason::NonFiniteSelectedTarget:
        return "non_finite_selected_target";
    case ReservoirRejectReason::NonPositiveSelectedTarget:
        return "non_positive_selected_target";
    case ReservoirRejectReason::NonFiniteUnbiasedContributionWeight:
        return "non_finite_unbiased_contribution_weight";
    case ReservoirRejectReason::SourceReservoirInvalid:
        return "source_reservoir_invalid";
    case ReservoirRejectReason::ShiftUnsupported:
        return "shift_unsupported";
    case ReservoirRejectReason::ShiftDestinationOutsideSupport:
        return "shift_destination_outside_support";
    case ReservoirRejectReason::NonFiniteJacobian:
        return "non_finite_jacobian";
    case ReservoirRejectReason::NonPositiveJacobian:
        return "non_positive_jacobian";
    case ReservoirRejectReason::NonFiniteMisWeight:
        return "non_finite_mis_weight";
    case ReservoirRejectReason::NegativeMisWeight:
        return "negative_mis_weight";
    case ReservoirRejectReason::ZeroMisWeight:
        return "zero_mis_weight";
    case ReservoirRejectReason::Count:
        break;
    }
    return "unknown";
}

void ReservoirValidationCounters::record(
    const ReservoirOperationResult &result) noexcept {
    if (result.accepted()) {
        ++accepted_operations;
    } else {
        ++rejected_operations;
        const std::size_t index =
            static_cast<std::size_t>(result.rejection);
        if (index < rejection_counts.size()) {
            ++rejection_counts[index];
        }
    }
    selected_operations += result.selected != 0u ? 1u : 0u;
    represented_candidates += result.represented_candidates;
}

std::uint64_t
ReservoirValidationCounters::count(ReservoirRejectReason reason) const noexcept {
    const std::size_t index = static_cast<std::size_t>(reason);
    return index < rejection_counts.size() ? rejection_counts[index] : 0u;
}

} // namespace restir

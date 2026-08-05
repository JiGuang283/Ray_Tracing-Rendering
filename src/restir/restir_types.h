#ifndef RESTIR_TYPES_H
#define RESTIR_TYPES_H

#include "host_device.h"

#include <cstdint>
#include <type_traits>

namespace restir {

enum ReservoirFlagBits : std::uint32_t {
    RESERVOIR_FLAG_NONE = 0,
    RESERVOIR_FLAG_HAS_SAMPLE = 1u << 0u,
    RESERVOIR_FLAG_FINALIZED = 1u << 1u,
};

enum CandidateFlagBits : std::uint32_t {
    CANDIDATE_FLAG_NONE = 0,
    CANDIDATE_FLAG_VALID_SAMPLE = 1u << 0u,
    CANDIDATE_FLAG_NONZERO_CONTRIBUTION = 1u << 1u,
};

enum class ReservoirRejectReason : std::uint32_t {
    None = 0,
    InvalidSample,
    ReservoirFinalized,
    InvalidReservoirState,
    NonFiniteRandom,
    RandomOutOfRange,
    NonFiniteTarget,
    NegativeTarget,
    ZeroTarget,
    ZeroTargetWithContribution,
    NonFiniteProposalPdf,
    NonPositiveProposalPdf,
    ZeroProposalWithContribution,
    NonFiniteWeight,
    NonPositiveWeight,
    NonFiniteWeightSum,
    CandidateCountOverflow,
    EmptyReservoir,
    NoSelectedSample,
    NonFiniteSelectedTarget,
    NonPositiveSelectedTarget,
    NonFiniteUnbiasedContributionWeight,
    SourceReservoirInvalid,
    ShiftUnsupported,
    ShiftDestinationOutsideSupport,
    NonFiniteJacobian,
    NonPositiveJacobian,
    NonFiniteMisWeight,
    NegativeMisWeight,
    ZeroMisWeight,
    Count,
};

enum class RestirShiftFailure : std::uint32_t {
    None = 0,
    Unsupported,
    DestinationOutsideSupport,
    InvalidSample,
    NonFiniteDestinationTarget,
    NonPositiveDestinationTarget,
    NonFiniteJacobian,
    NonPositiveJacobian,
    NonFiniteMisWeight,
    NegativeMisWeight,
};

template <typename Sample, typename Scalar>
struct alignas(16) ReservoirCandidate {
    Sample sample{};
    Scalar target = Scalar(0);
    Scalar proposal_pdf = Scalar(0);
    std::uint32_t flags = CANDIDATE_FLAG_NONE;
};

template <typename Sample, typename Scalar>
struct alignas(16) Reservoir {
    Sample sample{};
    // Raw resampling mass. It is never overwritten with the finalized UCW.
    Scalar weight_sum = Scalar(0);
    Scalar selected_target = Scalar(0);
    Scalar unbiased_contribution_weight = Scalar(0);
    // Number of represented proposal draws, including valid zero-target draws.
    std::uint32_t M = 0;
    std::uint32_t flags = RESERVOIR_FLAG_NONE;
};

struct alignas(16) ReservoirOperationResult {
    ReservoirRejectReason rejection = ReservoirRejectReason::None;
    std::uint32_t selected = 0;
    std::uint32_t represented_candidates = 0;
    std::uint32_t reserved = 0;

    RT_HOST_DEVICE bool accepted() const noexcept {
        return rejection == ReservoirRejectReason::None;
    }

    RT_HOST_DEVICE bool changed_selection() const noexcept {
        return selected != 0;
    }
};

template <typename Scalar> struct alignas(16) RestirShiftResult {
    // |d(destination sample) / d(source sample)| in the declared measure.
    Scalar jacobian = Scalar(0);
    Scalar destination_target = Scalar(0);
    Scalar mis_weight = Scalar(0);
    std::uint32_t supported = 0;
    RestirShiftFailure failure = RestirShiftFailure::Unsupported;
};

template <typename Sample, typename Scalar>
RT_HOST_DEVICE ReservoirCandidate<Sample, Scalar>
make_candidate(const Sample &sample, Scalar target, Scalar proposal_pdf,
               bool contribution_nonzero = true,
               bool sample_valid = true) noexcept {
    ReservoirCandidate<Sample, Scalar> candidate;
    candidate.sample = sample;
    candidate.target = target;
    candidate.proposal_pdf = proposal_pdf;
    candidate.flags = sample_valid ? CANDIDATE_FLAG_VALID_SAMPLE
                                   : CANDIDATE_FLAG_NONE;
    if (contribution_nonzero) {
        candidate.flags |= CANDIDATE_FLAG_NONZERO_CONTRIBUTION;
    }
    return candidate;
}

static_assert(sizeof(ReservoirOperationResult) == 16,
              "Reservoir operation result ABI changed");
static_assert(sizeof(ReservoirCandidate<std::uint32_t, float>) == 16,
              "Float candidate ABI changed");
static_assert(sizeof(ReservoirCandidate<std::uint32_t, double>) == 32,
              "Double candidate ABI changed");
static_assert(sizeof(Reservoir<std::uint32_t, float>) == 32,
              "Float reservoir ABI changed");
static_assert(sizeof(Reservoir<std::uint32_t, double>) == 48,
              "Double reservoir ABI changed");
static_assert(sizeof(RestirShiftResult<float>) == 32,
              "Float shift result ABI changed");
static_assert(sizeof(RestirShiftResult<double>) == 32,
              "Double shift result ABI changed");
static_assert(
    std::is_trivially_copyable_v<Reservoir<std::uint32_t, float>>,
    "Device reservoirs must be trivially copyable");
static_assert(
    std::is_trivially_copyable_v<RestirShiftResult<float>>,
    "Device shift results must be trivially copyable");

} // namespace restir

#endif

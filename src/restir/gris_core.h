#ifndef RESTIR_GRIS_CORE_H
#define RESTIR_GRIS_CORE_H

#include "reservoir_core.h"

namespace restir {
namespace detail {

RT_HOST_DEVICE RT_FORCE_INLINE ReservoirRejectReason
shift_failure_to_rejection(RestirShiftFailure failure) noexcept {
    switch (failure) {
    case RestirShiftFailure::None:
        return ReservoirRejectReason::None;
    case RestirShiftFailure::Unsupported:
        return ReservoirRejectReason::ShiftUnsupported;
    case RestirShiftFailure::DestinationOutsideSupport:
        return ReservoirRejectReason::ShiftDestinationOutsideSupport;
    case RestirShiftFailure::InvalidSample:
        return ReservoirRejectReason::InvalidSample;
    case RestirShiftFailure::NonFiniteDestinationTarget:
        return ReservoirRejectReason::NonFiniteTarget;
    case RestirShiftFailure::NonPositiveDestinationTarget:
        return ReservoirRejectReason::NonPositiveSelectedTarget;
    case RestirShiftFailure::NonFiniteJacobian:
        return ReservoirRejectReason::NonFiniteJacobian;
    case RestirShiftFailure::NonPositiveJacobian:
        return ReservoirRejectReason::NonPositiveJacobian;
    case RestirShiftFailure::NonFiniteMisWeight:
        return ReservoirRejectReason::NonFiniteMisWeight;
    case RestirShiftFailure::NegativeMisWeight:
        return ReservoirRejectReason::NegativeMisWeight;
    }
    return ReservoirRejectReason::ShiftUnsupported;
}

} // namespace detail

template <typename Scalar>
RT_HOST_DEVICE RT_FORCE_INLINE RestirShiftResult<Scalar>
identity_shift(Scalar destination_target,
               Scalar mis_weight = Scalar(1)) noexcept {
    RestirShiftResult<Scalar> result;
    result.jacobian = Scalar(1);
    result.destination_target = destination_target;
    result.mis_weight = mis_weight;
    result.supported = 1u;
    result.failure = RestirShiftFailure::None;
    return result;
}

template <typename Scalar>
RT_HOST_DEVICE RT_FORCE_INLINE RestirShiftResult<Scalar>
failed_shift(RestirShiftFailure failure) noexcept {
    RestirShiftResult<Scalar> result;
    result.failure = failure;
    return result;
}

template <typename Scalar>
RT_HOST_DEVICE RT_FORCE_INLINE bool compute_balance_mis_weight(
    Scalar numerator, Scalar denominator, Scalar &weight) noexcept {
    weight = Scalar(0);
    if (!detail::finite(numerator) || !detail::finite(denominator) ||
        numerator < Scalar(0) || denominator <= Scalar(0) ||
        numerator > denominator) {
        return false;
    }
    weight = numerator / denominator;
    return detail::finite(weight) && weight >= Scalar(0) &&
           weight <= Scalar(1);
}

template <typename Sample, typename Scalar>
RT_HOST_DEVICE RT_FORCE_INLINE ReservoirOperationResult combine_reservoir(
    Reservoir<Sample, Scalar> &destination,
    const Reservoir<Sample, Scalar> &source, const Sample &shifted_sample,
    const RestirShiftResult<Scalar> &shift, Scalar random,
    bool shifted_sample_valid = true) noexcept {
    if (!shifted_sample_valid) {
        return detail::operation_result(ReservoirRejectReason::InvalidSample);
    }
    if ((destination.flags & RESERVOIR_FLAG_FINALIZED) != 0u) {
        return detail::operation_result(
            ReservoirRejectReason::ReservoirFinalized);
    }
    if (!detail::raw_state_valid(destination)) {
        return detail::operation_result(
            ReservoirRejectReason::InvalidReservoirState);
    }
    if (!reservoir_is_usable(source)) {
        return detail::operation_result(
            ReservoirRejectReason::SourceReservoirInvalid);
    }
    if (shift.supported == 0u ||
        shift.failure != RestirShiftFailure::None) {
        const ReservoirRejectReason rejection =
            shift.failure == RestirShiftFailure::None
                ? ReservoirRejectReason::ShiftUnsupported
                : detail::shift_failure_to_rejection(shift.failure);
        return detail::operation_result(rejection);
    }
    if (!detail::finite(shift.destination_target)) {
        return detail::operation_result(ReservoirRejectReason::NonFiniteTarget);
    }
    if (shift.destination_target <= Scalar(0)) {
        return detail::operation_result(
            ReservoirRejectReason::ShiftDestinationOutsideSupport);
    }
    if (!detail::finite(shift.jacobian)) {
        return detail::operation_result(
            ReservoirRejectReason::NonFiniteJacobian);
    }
    if (shift.jacobian <= Scalar(0)) {
        return detail::operation_result(
            ReservoirRejectReason::NonPositiveJacobian);
    }
    if (!detail::finite(shift.mis_weight)) {
        return detail::operation_result(
            ReservoirRejectReason::NonFiniteMisWeight);
    }
    if (shift.mis_weight < Scalar(0)) {
        return detail::operation_result(ReservoirRejectReason::NegativeMisWeight);
    }
    if (destination.M >
        std::numeric_limits<std::uint32_t>::max() - source.M) {
        return detail::operation_result(
            ReservoirRejectReason::CandidateCountOverflow);
    }

    if (shift.mis_weight == Scalar(0)) {
        destination.M += source.M;
        return detail::operation_result(ReservoirRejectReason::ZeroMisWeight,
                                        false, source.M);
    }

    // Reconstruct the source stream's mass at the destination. The source
    // UCW is W_s = weight_sum_s / (M_s * target_s); destination target and
    // the forward shift Jacobian then convert that represented stream.
    Scalar weight = source.unbiased_contribution_weight *
                    static_cast<Scalar>(source.M);
    weight *= shift.destination_target;
    weight *= shift.jacobian;
    weight *= shift.mis_weight;
    if (!detail::finite(weight)) {
        return detail::operation_result(ReservoirRejectReason::NonFiniteWeight);
    }
    if (weight <= Scalar(0)) {
        return detail::operation_result(
            ReservoirRejectReason::NonPositiveWeight);
    }
    return detail::stream_weight(destination, shifted_sample,
                                 shift.destination_target, weight, source.M,
                                 random);
}

} // namespace restir

#endif

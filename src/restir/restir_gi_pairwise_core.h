#ifndef RESTIR_GI_PAIRWISE_CORE_H
#define RESTIR_GI_PAIRWISE_CORE_H

#include "restir_gi_core.h"
#include "restir_spatial_pairwise_core.h"

namespace restir {

RT_HOST_DEVICE RT_FORCE_INLINE RestirGIStatus evaluate_gi_pairwise_target(
    const RestirDIPixelContext &context, const RestirGISample &sample,
    float &target, RestirGIShiftFailure &failure) noexcept {
    RestirGIReconnectResult reconnect;
    const RestirGIStatus status =
        evaluate_diffuse_reconnection(context, sample, reconnect);
    target = reconnect.target;
    failure = reconnect.failure;
    return status;
}

RT_HOST_DEVICE RT_FORCE_INLINE RestirGIStatus stream_pairwise_gi_mass(
    RestirGIReservoir &output, const RestirGISample &sample,
    float destination_target, float weight,
    std::uint32_t represented_count, float effective_count, float random,
    bool sample_valid, bool &changed_selection) noexcept {
    changed_selection = false;
    if (!sample_valid || !(destination_target > 0.0f) || !(weight > 0.0f)) {
        const ReservoirOperationResult represented =
            represent_gi_candidates(output, represented_count,
                                    effective_count);
        return represented.accepted() ||
                       represented.rejection ==
                           ReservoirRejectReason::ZeroTarget
                   ? RestirGIStatus::Success
                   : RestirGIStatus::ReservoirFailure;
    }
    const ReservoirOperationResult streamed = stream_gi_weight(
        output, sample, destination_target, weight, represented_count,
        effective_count, random);
    changed_selection = streamed.changed_selection();
    return streamed.accepted() ? RestirGIStatus::Success
                               : RestirGIStatus::ReservoirFailure;
}

} // namespace restir

#endif

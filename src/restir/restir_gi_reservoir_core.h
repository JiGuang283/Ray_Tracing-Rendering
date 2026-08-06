#ifndef RESTIR_GI_RESERVOIR_CORE_H
#define RESTIR_GI_RESERVOIR_CORE_H

#include "restir_gi_types.h"
#include "restir_specialized_reservoir_core.h"

namespace restir {

RT_HOST_DEVICE RT_FORCE_INLINE void
reset_reservoir(RestirGIReservoir &reservoir) noexcept {
    specialized_reservoir::reset(reservoir);
}

RT_HOST_DEVICE RT_FORCE_INLINE bool
reservoir_has_sample(const RestirGIReservoir &reservoir) noexcept {
    return specialized_reservoir::has_sample(reservoir);
}

RT_HOST_DEVICE RT_FORCE_INLINE bool
reservoir_is_finalized(const RestirGIReservoir &reservoir) noexcept {
    return specialized_reservoir::is_finalized(reservoir);
}

RT_HOST_DEVICE RT_FORCE_INLINE bool
reservoir_is_usable(const RestirGIReservoir &reservoir) noexcept {
    return specialized_reservoir::is_usable(reservoir);
}

RT_HOST_DEVICE RT_FORCE_INLINE ReservoirOperationResult stream_gi_weight(
    RestirGIReservoir &reservoir, const RestirGISample &sample,
    float selected_target, float weight, std::uint32_t represented_count,
    float effective_count, float random) noexcept {
    return specialized_reservoir::stream_weight(
        reservoir, sample, selected_target, weight, represented_count,
        effective_count, random);
}

RT_HOST_DEVICE RT_FORCE_INLINE ReservoirOperationResult
represent_gi_candidates(RestirGIReservoir &reservoir,
                        std::uint32_t represented_count,
                        float effective_count) noexcept {
    return specialized_reservoir::represent_candidates(
        reservoir, represented_count, effective_count);
}

RT_HOST_DEVICE RT_FORCE_INLINE ReservoirOperationResult stream_candidate(
    RestirGIReservoir &reservoir, const RestirGICandidate &candidate,
    float random) noexcept {
    return specialized_reservoir::stream_candidate(reservoir, candidate,
                                                    random);
}

RT_HOST_DEVICE RT_FORCE_INLINE ReservoirOperationResult
finalize_gi_reservoir(RestirGIReservoir &reservoir,
                      float normalization_denominator) noexcept {
    return specialized_reservoir::finalize(reservoir,
                                           normalization_denominator);
}

RT_HOST_DEVICE RT_FORCE_INLINE ReservoirOperationResult
finalize_reservoir(RestirGIReservoir &reservoir) noexcept {
    return finalize_gi_reservoir(reservoir, reservoir.effective_M);
}

} // namespace restir

#endif

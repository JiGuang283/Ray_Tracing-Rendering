#ifndef RESTIR_DI_TYPES_H
#define RESTIR_DI_TYPES_H

#include "restir_types.h"
#include "packed_types.h"

#include <cstdint>
#include <type_traits>

namespace restir {

struct alignas(16) RestirLightSample {
    std::uint32_t light_id = kInvalidPackedIndex;
    std::uint32_t element_id = kInvalidPackedIndex;
    std::uint32_t type = 0;
    std::uint32_t flags = 0;
    Float4 canonical_data{};

    RT_HOST_DEVICE bool valid() const noexcept {
        return light_id != kInvalidPackedIndex;
    }
};

using RestirDICandidate = ReservoirCandidate<RestirLightSample, float>;

struct alignas(16) RestirDIReservoir {
    RestirLightSample sample{};
    // Raw resampling mass is retained after finalization for diagnostics and
    // later resampling. effective_M may be fractional under pairwise MIS.
    float weight_sum = 0.0f;
    float selected_target = 0.0f;
    float unbiased_contribution_weight = 0.0f;
    float effective_M = 0.0f;
    // Integer proposal count is kept separately from effective_M so statistics
    // retain their literal meaning.
    std::uint32_t M = 0;
    std::uint32_t flags = RESERVOIR_FLAG_NONE;
    std::uint32_t age = 0;
    std::uint32_t reserved = 0;
};

static_assert(sizeof(RestirLightSample) == 32,
              "canonical ReSTIR light samples must remain 32 bytes");
static_assert(sizeof(RestirDIReservoir) == 64,
              "initial ReSTIR DI reservoirs must remain 64 bytes");
static_assert(std::is_trivially_copyable_v<RestirLightSample>);
static_assert(std::is_trivially_copyable_v<RestirDIReservoir>);

} // namespace restir

#endif

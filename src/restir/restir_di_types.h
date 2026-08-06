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
using RestirDIReservoir = Reservoir<RestirLightSample, float>;

static_assert(sizeof(RestirLightSample) == 32,
              "canonical ReSTIR light samples must remain 32 bytes");
static_assert(sizeof(RestirDIReservoir) == 64,
              "initial ReSTIR DI reservoirs must remain 64 bytes");
static_assert(std::is_trivially_copyable_v<RestirLightSample>);
static_assert(std::is_trivially_copyable_v<RestirDIReservoir>);

} // namespace restir

#endif

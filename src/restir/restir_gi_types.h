#ifndef RESTIR_GI_TYPES_H
#define RESTIR_GI_TYPES_H

#include "packed_types.h"
#include "reservoir_core.h"

#include <cstdint>
#include <type_traits>

namespace restir {

// A diffuse GI sample is canonicalized to world-space area measure at x2.
// The cached suffix radiance is valid only for direction-independent diffuse
// scattering at the secondary vertex.
struct alignas(16) RestirGISample {
    Float3 position{};
    float source_pdf_area = 0.0f;

    Float3 suffix_radiance{};
    std::uint32_t geometric_normal = 0u;

    std::uint32_t material_id = kInvalidPackedIndex;
    std::uint32_t instance_id = kInvalidPackedIndex;
    std::uint32_t primitive_id = kInvalidPackedIndex;
    std::uint32_t source_pixel = kInvalidPackedIndex;

    RT_HOST_DEVICE bool valid() const noexcept {
        return instance_id != kInvalidPackedIndex &&
               primitive_id != kInvalidPackedIndex;
    }
};

using RestirGICandidate = ReservoirCandidate<RestirGISample, float>;

struct alignas(16) RestirGIReservoir {
    RestirGISample sample{};
    float weight_sum = 0.0f;
    float selected_target = 0.0f;
    float unbiased_contribution_weight = 0.0f;
    float effective_M = 0.0f;
    std::uint32_t M = 0u;
    std::uint32_t flags = RESERVOIR_FLAG_NONE;
    std::uint32_t age = 0u;
    std::uint32_t reserved = 0u;
};

static_assert(sizeof(RestirGISample) == 48,
              "diffuse ReSTIR GI samples must remain 48 bytes");
static_assert(sizeof(RestirGIReservoir) == 80,
              "diffuse ReSTIR GI reservoirs must remain 80 bytes");
static_assert(std::is_trivially_copyable_v<RestirGISample>);
static_assert(std::is_trivially_copyable_v<RestirGIReservoir>);

} // namespace restir

#endif

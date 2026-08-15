#ifndef RESTIR_GI_TYPES_H
#define RESTIR_GI_TYPES_H

#include "packed_types.h"
#include "reservoir_core.h"

#include <cstdint>
#include <type_traits>

namespace restir {

enum class RestirGIShiftMapping : std::uint32_t {
    Reconnect = 0u,
    RandomReplay = 1u,
};

inline constexpr float kRandomReplayProposalDensity = 1.0f;
inline constexpr float kRandomReplayJacobian = 1.0f;

// Reconnection samples are canonicalized to world-space area measure at x2.
// Random-replay samples identify a primary-sample-space sequence, whose
// Jacobian is one. The two domains are explicitly tagged and never inferred
// from payload values.
struct alignas(16) RestirGISample {
    Float3 position{};
    float source_pdf_area = 0.0f;

    Float3 suffix_radiance{};
    std::uint32_t geometric_normal = 0u;

    std::uint32_t material_id = kInvalidPackedIndex;
    std::uint32_t instance_id = kInvalidPackedIndex;
    std::uint32_t primitive_id = kInvalidPackedIndex;
    std::uint32_t source_pixel = kInvalidPackedIndex;

    RestirGIShiftMapping mapping = RestirGIShiftMapping::Reconnect;
    std::uint32_t replay_seed = 0u;
    std::uint32_t path_length = 0u;
    std::uint32_t reserved = 0u;

    RT_HOST_DEVICE bool valid() const noexcept {
        if (mapping == RestirGIShiftMapping::RandomReplay) {
            return replay_seed != 0u &&
                   source_pixel != kInvalidPackedIndex;
        }
        return mapping == RestirGIShiftMapping::Reconnect &&
               instance_id != kInvalidPackedIndex &&
               primitive_id != kInvalidPackedIndex;
    }

    RT_HOST_DEVICE bool reconnect() const noexcept {
        return mapping == RestirGIShiftMapping::Reconnect;
    }

    RT_HOST_DEVICE bool random_replay() const noexcept {
        return mapping == RestirGIShiftMapping::RandomReplay;
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

static_assert(sizeof(RestirGISample) == 64,
              "hybrid ReSTIR GI samples must remain 64 bytes");
static_assert(sizeof(RestirGIReservoir) == 96,
              "hybrid ReSTIR GI reservoirs must remain 96 bytes");
static_assert(std::is_trivially_copyable_v<RestirGISample>);
static_assert(std::is_trivially_copyable_v<RestirGIReservoir>);

} // namespace restir

#endif

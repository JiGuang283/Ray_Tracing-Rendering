#ifndef RESTIR_VALIDATION_H
#define RESTIR_VALIDATION_H

#include "restir_types.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace restir {

constexpr std::size_t reservoir_reject_reason_count =
    static_cast<std::size_t>(ReservoirRejectReason::Count);

const char *reservoir_reject_reason_name(
    ReservoirRejectReason reason) noexcept;

struct ReservoirValidationCounters {
    std::uint64_t accepted_operations = 0;
    std::uint64_t rejected_operations = 0;
    std::uint64_t selected_operations = 0;
    std::uint64_t represented_candidates = 0;
    std::array<std::uint64_t, reservoir_reject_reason_count> rejection_counts{};

    void record(const ReservoirOperationResult &result) noexcept;
    std::uint64_t count(ReservoirRejectReason reason) const noexcept;
};

} // namespace restir

#endif

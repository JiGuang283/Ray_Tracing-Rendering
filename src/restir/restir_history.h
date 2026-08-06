#ifndef RESTIR_HISTORY_H
#define RESTIR_HISTORY_H

#include "render_session.h"

#include <cstdint>
#include <type_traits>

namespace restir {

constexpr std::uint32_t kInvalidHistoryBuffer = 0xffffffffu;

enum class RestirHistoryResetReason : std::uint32_t {
    None = 0,
    Explicit,
    Uninitialized,
    ExtentChanged,
    IntegratorChanged,
    SettingsChanged,
    CameraRevisionChanged,
    GeometryRevisionChanged,
    MaterialRevisionChanged,
    LightingRevisionChanged,
    FrameIndexRewound,
    Count,
};

struct alignas(16) RestirHistoryKey {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    IntegratorKind integrator = IntegratorKind::ReSTIRDI;
    std::uint32_t reserved = 0;
    std::uint64_t settings = 0;
    SceneRevision revision;
};

struct alignas(16) RestirFrameState {
    RestirHistoryKey key;
    std::uint64_t completed_iterations = 0;
    std::uint64_t history_generation = 0;
    std::uint64_t last_frame_index = 0;
    std::uint32_t committed_buffer = 0;
    std::uint32_t history_valid = 0;
    std::uint32_t reserved[2]{};
};

struct RestirFramePreparation {
    RestirHistoryResetReason reset_reason =
        RestirHistoryResetReason::None;
    std::uint32_t read_buffer = kInvalidHistoryBuffer;
    std::uint32_t write_buffer = 0;

    bool reset() const noexcept {
        return reset_reason != RestirHistoryResetReason::None;
    }
};

RestirHistoryKey make_restir_history_key(
    const RenderFrameRequest &request) noexcept;
RestirHistoryResetReason compare_restir_history(
    const RestirHistoryKey &previous,
    const RestirHistoryKey &current) noexcept;
RestirFramePreparation prepare_restir_frame(
    RestirFrameState &state, const RenderFrameRequest &request);
void commit_restir_iteration(RestirFrameState &state,
                             std::uint32_t write_buffer,
                             std::uint64_t frame_index);
void reset_restir_history(RestirFrameState &state) noexcept;
const char *restir_history_reset_reason_name(
    RestirHistoryResetReason reason) noexcept;

static_assert(std::is_trivially_copyable_v<RestirHistoryKey>);
static_assert(std::is_trivially_copyable_v<RestirFrameState>);
static_assert(static_cast<std::uint32_t>(
                  RestirHistoryResetReason::Count) <=
              kRestirHistoryFailureBuckets);

} // namespace restir

#endif

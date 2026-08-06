#include "restir_history.h"

#include <stdexcept>

namespace restir {
namespace {

void apply_reset(RestirFrameState &state, const RestirHistoryKey &key) {
    const std::uint64_t generation = state.history_generation + 1u;
    state = {};
    state.key = key;
    state.history_generation = generation;
}

} // namespace

RestirHistoryKey make_restir_history_key(
    const RenderFrameRequest &request) noexcept {
    RestirHistoryKey key;
    key.width = request.render.extent.width;
    key.height = request.render.extent.height;
    key.integrator = request.render.integrator;
    key.settings = restir_settings_fingerprint(request.render.restir);
    key.revision = request.revision;
    return key;
}

RestirHistoryResetReason compare_restir_history(
    const RestirHistoryKey &previous,
    const RestirHistoryKey &current) noexcept {
    if (previous.width != current.width ||
        previous.height != current.height) {
        return RestirHistoryResetReason::ExtentChanged;
    }
    if (previous.integrator != current.integrator) {
        return RestirHistoryResetReason::IntegratorChanged;
    }
    if (previous.settings != current.settings) {
        return RestirHistoryResetReason::SettingsChanged;
    }
    if (previous.revision.camera != current.revision.camera) {
        return RestirHistoryResetReason::CameraRevisionChanged;
    }
    if (previous.revision.geometry != current.revision.geometry) {
        return RestirHistoryResetReason::GeometryRevisionChanged;
    }
    if (previous.revision.material != current.revision.material) {
        return RestirHistoryResetReason::MaterialRevisionChanged;
    }
    if (previous.revision.lighting != current.revision.lighting) {
        return RestirHistoryResetReason::LightingRevisionChanged;
    }
    return RestirHistoryResetReason::None;
}

RestirFramePreparation prepare_restir_frame(
    RestirFrameState &state, const RenderFrameRequest &request) {
    validate_render_frame_request(request);
    const RestirHistoryKey key = make_restir_history_key(request);
    RestirHistoryResetReason reset_reason =
        RestirHistoryResetReason::None;
    if (request.render.restir.history_mode == RestirHistoryMode::Reset) {
        reset_reason = RestirHistoryResetReason::Explicit;
    } else if (state.history_valid == 0u) {
        reset_reason = RestirHistoryResetReason::Uninitialized;
    } else {
        reset_reason = compare_restir_history(state.key, key);
        if (reset_reason == RestirHistoryResetReason::None &&
            request.frame_index < state.last_frame_index) {
            reset_reason = RestirHistoryResetReason::FrameIndexRewound;
        }
    }
    if (reset_reason != RestirHistoryResetReason::None) {
        apply_reset(state, key);
    } else {
        state.key = key;
    }

    RestirFramePreparation preparation;
    preparation.reset_reason = reset_reason;
    preparation.read_buffer = state.history_valid != 0u
                                  ? state.committed_buffer
                                  : kInvalidHistoryBuffer;
    preparation.write_buffer = state.history_valid != 0u
                                   ? state.committed_buffer ^ 1u
                                   : 0u;
    return preparation;
}

void commit_restir_iteration(RestirFrameState &state,
                             std::uint32_t write_buffer,
                             std::uint64_t frame_index) {
    if (write_buffer > 1u) {
        throw std::invalid_argument(
            "ReSTIR history buffer index must be 0 or 1");
    }
    state.committed_buffer = write_buffer;
    state.history_valid = 1u;
    ++state.completed_iterations;
    state.last_frame_index = frame_index;
}

void reset_restir_history(RestirFrameState &state) noexcept {
    const std::uint64_t generation = state.history_generation + 1u;
    state = {};
    state.history_generation = generation;
}

const char *restir_history_reset_reason_name(
    RestirHistoryResetReason reason) noexcept {
    switch (reason) {
    case RestirHistoryResetReason::None:
        return "none";
    case RestirHistoryResetReason::Explicit:
        return "explicit";
    case RestirHistoryResetReason::Uninitialized:
        return "uninitialized";
    case RestirHistoryResetReason::ExtentChanged:
        return "extent_changed";
    case RestirHistoryResetReason::IntegratorChanged:
        return "integrator_changed";
    case RestirHistoryResetReason::SettingsChanged:
        return "settings_changed";
    case RestirHistoryResetReason::CameraRevisionChanged:
        return "camera_revision_changed";
    case RestirHistoryResetReason::GeometryRevisionChanged:
        return "geometry_revision_changed";
    case RestirHistoryResetReason::MaterialRevisionChanged:
        return "material_revision_changed";
    case RestirHistoryResetReason::LightingRevisionChanged:
        return "lighting_revision_changed";
    case RestirHistoryResetReason::FrameIndexRewound:
        return "frame_index_rewound";
    case RestirHistoryResetReason::Count:
        break;
    }
    return "unknown";
}

} // namespace restir

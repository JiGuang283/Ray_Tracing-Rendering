#ifndef CUDA_RESTIR_SCHEDULER_H
#define CUDA_RESTIR_SCHEDULER_H

#include "restir_workspace.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <vector>

namespace cuda_backend {

struct CudaRestirSkeletonSettings {
    RenderFrameRequest frame;
    PackedTransportSettings reference_transport;
    std::uint32_t block_size = 128;
};

struct CudaRestirSchedulerStats {
    float milliseconds = 0.0f;
    std::uint64_t completed_iterations = 0;
    std::uint64_t sample_count = 0;
    std::uint64_t traversal_steps = 0;
    std::uint64_t shadow_rays = 0;
    std::uint64_t clamped_samples = 0;
    std::uint64_t invalid_samples = 0;
    std::array<std::uint64_t, 7> gbuffer_status{};
    std::array<std::uint64_t, 8> transport_status{};
    restir::RestirHistoryResetReason history_reset_reason =
        restir::RestirHistoryResetReason::None;
    CudaRestirWorkspaceInfo workspace;
    bool cancelled = false;
};

struct CudaRestirSchedulerOutput {
    std::vector<CudaFilmPixel> film;
    std::vector<restir::RestirSurface> gbuffer;
    CudaRestirSchedulerStats stats;
};

CudaRestirSchedulerOutput render_restir_skeleton_cuda(
    DeviceSceneView scene, const CudaRestirSkeletonSettings &settings,
    CudaRestirWorkspace &workspace,
    const std::atomic<bool> *cancel = nullptr);

} // namespace cuda_backend

#endif

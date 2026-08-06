#ifndef CUDA_RESTIR_SCHEDULER_H
#define CUDA_RESTIR_SCHEDULER_H

#include "restir_workspace.h"
#include "restir_di_types.h"

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
    std::array<std::uint64_t, 11> di_generation_status{};
    std::array<std::uint64_t, 11> di_shading_status{};
    std::uint64_t initial_candidates = 0;
    std::uint64_t represented_candidates = 0;
    std::uint64_t rejected_candidates = 0;
    std::uint64_t visibility_rays = 0;
    std::uint64_t di_clamped_samples = 0;
    std::uint64_t di_invalid_samples = 0;
    restir::RestirHistoryResetReason history_reset_reason =
        restir::RestirHistoryResetReason::None;
    CudaRestirWorkspaceInfo workspace;
    bool cancelled = false;
};

struct CudaRestirSchedulerOutput {
    std::vector<CudaFilmPixel> film;
    std::vector<CudaFilmPixel> direct_film;
    std::vector<restir::RestirSurface> gbuffer;
    std::vector<restir::RestirDIReservoir> di_reservoirs;
    CudaRestirSchedulerStats stats;
};

CudaRestirSchedulerOutput render_restir_skeleton_cuda(
    DeviceSceneView scene, const CudaRestirSkeletonSettings &settings,
    CudaRestirWorkspace &workspace,
    const std::atomic<bool> *cancel = nullptr);

} // namespace cuda_backend

#endif

#ifndef CUDA_RESTIR_SCHEDULER_H
#define CUDA_RESTIR_SCHEDULER_H

#include "restir_device_types.h"
#include "restir_workspace.h"
#include "restir_di_types.h"
#include "restir_gi_types.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <vector>

namespace cuda_backend {

struct CudaRestirSkeletonSettings {
    RenderFrameRequest frame;
    PackedTransportSettings reference_transport;
    std::uint32_t block_size = 128;
    bool generate_reference = true;
    bool fused_stages = true;
    // None disables all device counter updates. Summary keeps scalar
    // sample/visibility/workload counters; Full additionally keeps every
    // status/compatibility/rejection bucket used by diagnostics.
    CudaRestirStatsLevel collect_stats = CudaRestirStatsLevel::Full;
};

struct CudaRestirSchedulerStats {
    float milliseconds = 0.0f;
    std::uint64_t kernel_launches = 0;
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
    std::array<std::uint64_t, 11> di_spatial_status{};
    std::array<std::uint64_t, 11> di_temporal_status{};
    std::array<std::uint64_t, 9> spatial_compatibility{};
    std::array<std::uint64_t, 16> temporal_rejection{};
    std::array<std::uint64_t, 16> gi_generation_status{};
    std::array<std::uint64_t, 16> gi_shading_status{};
    std::array<std::uint64_t, 16> gi_spatial_status{};
    std::array<std::uint64_t, 16> gi_temporal_status{};
    std::array<std::uint64_t, 9> gi_spatial_compatibility{};
    std::array<std::uint64_t, 16> gi_temporal_rejection{};
    std::array<std::uint64_t, 16> gi_shift_failures{};
    std::uint64_t initial_candidates = 0;
    std::uint64_t represented_candidates = 0;
    std::uint64_t rejected_candidates = 0;
    std::uint64_t spatial_candidates = 0;
    std::uint64_t spatial_accepted = 0;
    std::uint64_t spatial_rejected = 0;
    std::uint64_t pairwise_fallbacks = 0;
    std::uint64_t temporal_candidates = 0;
    std::uint64_t temporal_accepted = 0;
    std::uint64_t temporal_pairwise_fallbacks = 0;
    std::uint64_t valid_reservoirs = 0;
    double average_represented_M = 0.0;
    double average_effective_M = 0.0;
    double average_age = 0.0;
    std::uint64_t visibility_rays = 0;
    std::uint64_t di_clamped_samples = 0;
    std::uint64_t di_invalid_samples = 0;
    std::uint64_t gi_initial_candidates = 0;
    std::uint64_t gi_represented_candidates = 0;
    std::uint64_t gi_rejected_candidates = 0;
    std::uint64_t gi_spatial_candidates = 0;
    std::uint64_t gi_spatial_accepted = 0;
    std::uint64_t gi_spatial_pairwise_fallbacks = 0;
    std::uint64_t gi_temporal_candidates = 0;
    std::uint64_t gi_temporal_accepted = 0;
    std::uint64_t gi_temporal_pairwise_fallbacks = 0;
    std::uint64_t gi_valid_reservoirs = 0;
    double gi_average_represented_M = 0.0;
    double gi_average_effective_M = 0.0;
    double gi_average_age = 0.0;
    std::uint64_t gi_visibility_rays = 0;
    std::uint64_t gi_fallbacks = 0;
    std::uint64_t gi_replay_candidates = 0;
    std::uint64_t gi_replay_evaluations = 0;
    std::uint64_t gi_replay_shadow_rays = 0;
    std::uint64_t gi_replay_traversal_steps = 0;
    std::uint64_t gi_reconnect_selections = 0;
    std::uint64_t gi_replay_selections = 0;
    std::uint64_t gi_clamped_samples = 0;
    std::uint64_t gi_invalid_samples = 0;
    std::uint64_t gi_suffix_shadow_rays = 0;
    std::uint64_t gi_suffix_traversal_steps = 0;
    std::uint64_t gi_unique_source_pixels = 0;
    std::uint64_t gi_max_source_reuse = 0;
    double gi_average_source_reuse = 0.0;
    restir::RestirHistoryResetReason history_reset_reason =
        restir::RestirHistoryResetReason::None;
    CudaRestirWorkspaceInfo workspace;
    bool cancelled = false;
};

struct CudaRestirSchedulerOutput {
    std::vector<CudaFilmPixel> film;
    std::vector<CudaFilmPixel> direct_film;
    std::vector<CudaFilmPixel> indirect_film;
    std::vector<restir::RestirSurface> gbuffer;
    std::vector<restir::RestirDIReservoir> di_reservoirs;
    std::vector<restir::RestirGIReservoir> gi_reservoirs;
    CudaRestirSchedulerStats stats;
};

CudaRestirSchedulerOutput render_restir_skeleton_cuda(
    DeviceSceneView scene, const CudaRestirSkeletonSettings &settings,
    CudaRestirWorkspace &workspace,
    const std::atomic<bool> *cancel = nullptr);

} // namespace cuda_backend

#endif

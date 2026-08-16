#ifndef RENDER_TYPES_H
#define RENDER_TYPES_H

#include "color_pipeline_settings.h"
#include "integrator_policy.h"
#include "restir_settings.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

struct ImageExtent {
    std::uint32_t width = 0;
    std::uint32_t height = 0;

    std::size_t pixel_count() const;
};

ImageExtent make_image_extent(int width, int height);
ImageExtent make_image_extent(int width, double aspect_ratio);

struct RenderRequest {
    ImageExtent extent;
    IntegratorKind integrator = IntegratorKind::MISPath;
    std::uint32_t samples_per_pixel = 1;
    std::uint32_t max_depth = 50;
    std::uint32_t seed = 1337;
    std::uint32_t threads = 0;
    std::uint32_t cuda_batch_size = 0;
    std::uint32_t cuda_samples_per_launch = 0;
    double sample_clamp = 0.0;
    ColorPipelineSettings color_pipeline;
    RestirSettings restir;
};

void validate_render_request(const RenderRequest &request);

struct PreparationStats {
    double compile_seconds = 0.0;
    double upload_seconds = 0.0;
    std::size_t scene_bytes = 0;
};

struct RenderStatsBase {
    double seconds = 0.0;
    double compile_seconds = 0.0;
    double upload_seconds = 0.0;
    double resolve_seconds = 0.0;
    int width = 0;
    int height = 0;
    int samples_per_pixel = 0;
    std::uint64_t requested_samples = 0;
    std::uint64_t completed_samples = 0;
    std::uint64_t sample_count = 0;
    unsigned seed = 1337;
    std::uint64_t clamped_samples = 0;
    std::uint64_t invalid_samples = 0;
    std::string backend = "cpu";
    std::size_t scene_bytes = 0;
    bool cancelled = false;
};

struct CpuRenderStats {
    int threads = 0;
    std::uint64_t traversal_steps = 0;
    std::uint64_t shadow_rays = 0;
};

struct CudaRenderStats {
    double device_seconds = 0.0;
    std::string device_name = "host";
    std::size_t workspace_bytes = 0;
    std::uint64_t workspace_generation = 0;
    std::uint32_t workspace_pixel_capacity = 0;
    std::uint32_t workspace_path_capacity = 0;
    std::uint64_t traversal_steps = 0;
    std::uint64_t shadow_rays = 0;
    int batch_size = 0;
    int batch_count = 0;
    std::uint64_t wavefront_advance_launches = 0;
    std::uint64_t wavefront_active_path_steps = 0;
    std::uint32_t samples_per_launch = 0;
    std::array<std::uint64_t, 8> status_counts{};
};

struct RenderStats {
    RenderStatsBase base;
    CpuRenderStats cpu;
    CudaRenderStats cuda;
    RestirStats restir;
};

class CancellationToken {
  public:
    CancellationToken() = default;

    bool is_cancelled() const noexcept;
    const std::atomic<bool> *native_flag() const noexcept;

  private:
    explicit CancellationToken(std::shared_ptr<std::atomic<bool>> state);

    std::shared_ptr<std::atomic<bool>> m_state;
    friend class CancellationSource;
};

class CancellationSource {
  public:
    CancellationSource();

    CancellationToken token() const;
    void cancel() const noexcept;
    bool is_cancelled() const noexcept;

  private:
    std::shared_ptr<std::atomic<bool>> m_state;
};

std::uint32_t render_sample_seed(std::uint32_t base_seed,
                                 std::uint32_t pixel_index,
                                 std::uint32_t sample_index) noexcept;

#endif

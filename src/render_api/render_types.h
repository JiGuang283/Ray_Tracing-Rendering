#ifndef RENDER_TYPES_H
#define RENDER_TYPES_H

#include "color_pipeline_settings.h"
#include "host_device.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

enum class RenderBackend : std::uint32_t {
    CPU,
    CUDA
};

enum class IntegratorKind : std::uint32_t {
    Path = 0,
    RussianRoulette = 1,
    PBRPath = 2,
    DirectLighting = 3,
    MISPath = 4
};

enum IntegratorPolicyFlags : std::uint32_t {
    INTEGRATOR_POLICY_NONE = 0,
    INTEGRATOR_POLICY_DIRECT_LIGHTING = 1u << 0,
    INTEGRATOR_POLICY_MIS = 1u << 1,
    INTEGRATOR_POLICY_RUSSIAN_ROULETTE = 1u << 2
};

struct IntegratorPolicy {
    IntegratorKind kind = IntegratorKind::MISPath;
    std::uint32_t flags = INTEGRATOR_POLICY_NONE;
    std::uint32_t rr_start_depth = 3;
    float rr_min_survival = 0.05f;

    RT_HOST_DEVICE bool uses_direct_lighting() const noexcept {
        return (flags & INTEGRATOR_POLICY_DIRECT_LIGHTING) != 0;
    }
    RT_HOST_DEVICE bool uses_mis() const noexcept {
        return (flags & INTEGRATOR_POLICY_MIS) != 0;
    }
    RT_HOST_DEVICE bool uses_russian_roulette() const noexcept {
        return (flags & INTEGRATOR_POLICY_RUSSIAN_ROULETTE) != 0;
    }
};

RT_HOST_DEVICE inline bool
valid_integrator_policy(const IntegratorPolicy &policy) noexcept {
    constexpr std::uint32_t known_flags =
        INTEGRATOR_POLICY_DIRECT_LIGHTING | INTEGRATOR_POLICY_MIS |
        INTEGRATOR_POLICY_RUSSIAN_ROULETTE;
    return static_cast<std::uint32_t>(policy.kind) <=
               static_cast<std::uint32_t>(IntegratorKind::MISPath) &&
           (policy.flags & ~known_flags) == 0 &&
           policy.rr_min_survival >= 0.0f &&
           policy.rr_min_survival <= 0.95f;
}

IntegratorKind integrator_kind_from_id(int id);
int integrator_id(IntegratorKind kind) noexcept;
IntegratorPolicy integrator_policy(IntegratorKind kind);

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
    double sample_clamp = 0.0;
    ColorPipelineSettings color_pipeline;
};

void validate_render_request(const RenderRequest &request);

struct PreparationStats {
    double compile_seconds = 0.0;
    double upload_seconds = 0.0;
    std::size_t scene_bytes = 0;
};

struct RenderStats {
    double seconds = 0.0;
    double compile_seconds = 0.0;
    double upload_seconds = 0.0;
    double device_seconds = 0.0;
    double resolve_seconds = 0.0;
    int width = 0;
    int height = 0;
    int samples_per_pixel = 0;
    std::uint64_t requested_samples = 0;
    std::uint64_t completed_samples = 0;
    std::uint64_t sample_count = 0;
    unsigned seed = 1337;
    int threads = 0;
    std::uint64_t clamped_samples = 0;
    std::uint64_t invalid_samples = 0;
    std::string backend = "cpu";
    std::string device_name = "host";
    std::size_t scene_bytes = 0;
    std::size_t workspace_bytes = 0;
    std::uint64_t traversal_steps = 0;
    std::uint64_t shadow_rays = 0;
    int batch_size = 0;
    int batch_count = 0;
    std::array<std::uint64_t, 8> status_counts{};
    bool cancelled = false;
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

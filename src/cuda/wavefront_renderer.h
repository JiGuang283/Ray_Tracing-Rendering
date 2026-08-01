#ifndef CUDA_WAVEFRONT_RENDERER_H
#define CUDA_WAVEFRONT_RENDERER_H

#include "device_scene.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace cuda_backend {

struct alignas(16) CudaFilmPixel {
    Float3 radiance{};
    std::uint32_t sample_count = 0;
};

static_assert(sizeof(CudaFilmPixel) == 16);
static_assert(std::is_trivially_copyable_v<CudaFilmPixel>);

struct CudaRenderSettings {
    PackedTransportSettings transport;
    std::uint32_t width = 1;
    std::uint32_t height = 1;
    std::uint32_t samples_per_pixel = 1;
    std::uint32_t seed = 1337;
    std::uint32_t batch_size = 0;
    std::uint32_t block_size = 128;
    float sample_clamp = 0.0f;
};

struct CudaRenderStats {
    float milliseconds = 0.0f;
    std::uint64_t sample_count = 0;
    std::uint64_t traversal_steps = 0;
    std::uint64_t shadow_rays = 0;
    std::uint64_t clamped_samples = 0;
    std::uint64_t invalid_samples = 0;
    std::uint32_t batch_size = 0;
    std::uint32_t batch_count = 0;
    std::size_t workspace_bytes = 0;
    std::array<std::uint64_t, 8> status_counts{};
};

struct CudaRenderOutput {
    std::vector<CudaFilmPixel> film;
    CudaRenderStats stats;
};

CudaRenderOutput render_wavefront_cuda(
    DeviceSceneView scene, const CudaRenderSettings &settings);

} // namespace cuda_backend

#endif

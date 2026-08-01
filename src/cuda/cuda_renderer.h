#ifndef CUDA_RENDERER_H
#define CUDA_RENDERER_H

#include "render_data/compiled_scene.h"
#include "renderer/color_pipeline.h"
#include "renderer/render_buffer.h"
#include "renderer/renderer.h"

#include <atomic>
#include <cstdint>

namespace cuda_backend {

struct CudaRendererSettings {
    int integrator_id = 4;
    std::uint32_t max_depth = 50;
    std::uint32_t samples_per_pixel = 1;
    std::uint32_t seed = 1337;
    std::uint32_t batch_size = 0;
    float sample_clamp = 0.0f;
    ColorPipelineSettings color_pipeline;
};

RenderStats render_cuda(const CompiledScene &scene,
                        const CudaRendererSettings &settings,
                        RenderBuffer &target_buffer,
                        const std::atomic<bool> *cancel = nullptr);

} // namespace cuda_backend

#endif

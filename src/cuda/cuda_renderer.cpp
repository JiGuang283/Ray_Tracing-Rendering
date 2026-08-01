#include "cuda_renderer.h"

#include "device_scene.h"
#include "wavefront_renderer.h"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>

namespace cuda_backend {
namespace {

PackedIntegratorType packed_integrator(int integrator_id) {
    switch (integrator_id) {
    case 0:
        return PackedIntegratorType::Path;
    case 1:
        return PackedIntegratorType::RussianRoulette;
    case 2:
        return PackedIntegratorType::PBRPath;
    case 3:
        return PackedIntegratorType::DirectLighting;
    case 4:
    default:
        return PackedIntegratorType::MISPath;
    }
}

} // namespace

RenderStats render_cuda(const CompiledScene &scene,
                        const CudaRendererSettings &settings,
                        RenderBuffer &target_buffer,
                        const std::atomic<bool> *cancel) {
    std::string unavailable_reason;
    if (!cuda_device_available(&unavailable_reason)) {
        throw std::runtime_error("CUDA backend unavailable: " +
                                 unavailable_reason);
    }
    if (settings.max_depth == 0 || settings.samples_per_pixel == 0) {
        throw std::invalid_argument(
            "CUDA max depth and sample count must be positive");
    }

    const auto begin = std::chrono::steady_clock::now();
    DeviceSceneStorage device_scene;
    const DeviceSceneUploadStats upload = device_scene.upload(scene);
    CudaRenderSettings transport_settings;
    transport_settings.transport.integrator =
        packed_integrator(settings.integrator_id);
    transport_settings.transport.max_depth = settings.max_depth;
    transport_settings.transport.rr_start_depth = 3;
    transport_settings.width =
        static_cast<std::uint32_t>(target_buffer.width());
    transport_settings.height =
        static_cast<std::uint32_t>(target_buffer.height());
    transport_settings.samples_per_pixel = settings.samples_per_pixel;
    transport_settings.seed = settings.seed;
    transport_settings.batch_size = settings.batch_size;
    transport_settings.sample_clamp = settings.sample_clamp;
    const CudaRenderOutput output = render_wavefront_cuda(
        device_scene.view(), transport_settings, cancel);

    const ColorPipeline color_pipeline(settings.color_pipeline);
    for (std::uint32_t y = 0; y < transport_settings.height; ++y) {
        for (std::uint32_t x = 0; x < transport_settings.width; ++x) {
            const CudaFilmPixel &pixel =
                output.film[y * transport_settings.width + x];
            if (pixel.sample_count == 0) {
                continue;
            }
            target_buffer.set_pixel(
                static_cast<int>(x), static_cast<int>(y),
                color_pipeline.to_display(
                    color(pixel.radiance.x, pixel.radiance.y,
                          pixel.radiance.z),
                    static_cast<int>(pixel.sample_count)));
        }
    }
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = end - begin;

    RenderStats stats;
    stats.seconds = elapsed.count();
    stats.width = target_buffer.width();
    stats.height = target_buffer.height();
    stats.samples_per_pixel =
        static_cast<int>(settings.samples_per_pixel);
    stats.sample_count = static_cast<long long>(output.stats.sample_count);
    stats.seed = settings.seed;
    stats.threads = 0;
    stats.clamped_samples =
        static_cast<long long>(output.stats.clamped_samples);
    stats.invalid_samples =
        static_cast<long long>(output.stats.invalid_samples);
    stats.backend = "cuda";
    stats.device_name = cuda_device_name();
    stats.upload_seconds = upload.milliseconds / 1000.0;
    stats.device_seconds = output.stats.milliseconds / 1000.0;
    stats.scene_bytes = upload.bytes;
    stats.workspace_bytes = output.stats.workspace_bytes;
    stats.traversal_steps =
        static_cast<long long>(output.stats.traversal_steps);
    stats.shadow_rays = static_cast<long long>(output.stats.shadow_rays);
    stats.batch_size = static_cast<int>(output.stats.batch_size);
    stats.batch_count = static_cast<int>(output.stats.batch_count);
    std::cout << "Rendering finished in " << stats.seconds
              << " seconds (CUDA device " << stats.device_seconds
              << " seconds)." << std::endl;
    return stats;
}

} // namespace cuda_backend

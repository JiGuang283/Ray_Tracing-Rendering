#include "cuda_renderer.h"

#include "beauty_film.h"
#include "device_scene.h"
#include "scene_compiler.h"
#include "wavefront_renderer.h"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>

namespace cuda_backend {
namespace {

PackedIntegratorType packed_integrator(IntegratorKind integrator) {
    switch (integrator) {
    case IntegratorKind::Path:
        return PackedIntegratorType::Path;
    case IntegratorKind::RussianRoulette:
        return PackedIntegratorType::RussianRoulette;
    case IntegratorKind::PBRPath:
        return PackedIntegratorType::PBRPath;
    case IntegratorKind::DirectLighting:
        return PackedIntegratorType::DirectLighting;
    case IntegratorKind::MISPath:
        return PackedIntegratorType::MISPath;
    }
    throw std::invalid_argument("unsupported CUDA integrator");
}

} // namespace

class CudaRenderSession final : public IRenderSession {
  public:
    explicit CudaRenderSession(const SceneIR &ir) {
        std::string unavailable_reason;
        if (!cuda_device_available(&unavailable_reason)) {
            throw std::runtime_error("CUDA backend unavailable: " +
                                     unavailable_reason);
        }

        const auto compile_begin = std::chrono::steady_clock::now();
        CompiledScene scene = compile_scene(ir);
        const auto compile_end = std::chrono::steady_clock::now();
        m_preparation.compile_seconds =
            std::chrono::duration<double>(compile_end - compile_begin)
                .count();
        const DeviceSceneUploadStats upload = m_device_scene.upload(scene);
        m_preparation.upload_seconds = upload.milliseconds / 1000.0;
        m_preparation.scene_bytes = upload.bytes;
        m_device_name = cuda_device_name();
    }

    const PreparationStats &preparation_stats() const noexcept override {
        return m_preparation;
    }

    RenderResult render(const RenderRequest &request,
                        const CancellationToken &cancel,
                        PreviewSurface *preview) override {
        validate_render_request(request);
        const auto begin = std::chrono::steady_clock::now();
        CudaRenderSettings transport_settings;
        transport_settings.transport.integrator =
            packed_integrator(request.integrator);
        transport_settings.transport.max_depth = request.max_depth;
        transport_settings.transport.rr_start_depth = 3;
        transport_settings.width = request.extent.width;
        transport_settings.height = request.extent.height;
        transport_settings.samples_per_pixel = request.samples_per_pixel;
        transport_settings.seed = request.seed;
        transport_settings.batch_size = request.cuda_batch_size;
        transport_settings.sample_clamp =
            static_cast<float>(request.sample_clamp);
        const CudaRenderOutput output = render_wavefront_cuda(
            m_device_scene.view(), transport_settings,
            cancel.native_flag());

        BeautyFilm beauty(request.extent);
        for (std::uint32_t y = 0; y < transport_settings.height; ++y) {
            for (std::uint32_t x = 0; x < transport_settings.width; ++x) {
                const CudaFilmPixel &pixel =
                    output.film[y * transport_settings.width + x];
                beauty.set_pixel(
                    static_cast<int>(x), static_cast<int>(y),
                    color(pixel.radiance.x, pixel.radiance.y,
                          pixel.radiance.z),
                    pixel.sample_count);
            }
        }
        const auto resolve_begin = std::chrono::steady_clock::now();
        RenderBuffer display =
            resolve_beauty(beauty, request.color_pipeline);
        const auto resolve_end = std::chrono::steady_clock::now();
        if (preview != nullptr) {
            preview->publish(display);
        }
        const auto end = std::chrono::steady_clock::now();

        RenderStats stats;
        stats.seconds =
            std::chrono::duration<double>(end - begin).count();
        stats.compile_seconds = m_preparation.compile_seconds;
        stats.upload_seconds = m_preparation.upload_seconds;
        stats.resolve_seconds =
            std::chrono::duration<double>(resolve_end - resolve_begin)
                .count();
        stats.width = static_cast<int>(request.extent.width);
        stats.height = static_cast<int>(request.extent.height);
        stats.samples_per_pixel =
            static_cast<int>(request.samples_per_pixel);
        stats.requested_samples =
            static_cast<std::uint64_t>(request.extent.pixel_count()) *
            request.samples_per_pixel;
        stats.completed_samples = output.stats.sample_count;
        stats.sample_count = stats.completed_samples;
        stats.seed = request.seed;
        stats.threads = 0;
        stats.clamped_samples = output.stats.clamped_samples;
        stats.invalid_samples = output.stats.invalid_samples;
        stats.backend = "cuda";
        stats.device_name = m_device_name;
        stats.device_seconds = output.stats.milliseconds / 1000.0;
        stats.scene_bytes = m_preparation.scene_bytes;
        stats.workspace_bytes = output.stats.workspace_bytes;
        stats.traversal_steps = output.stats.traversal_steps;
        stats.shadow_rays = output.stats.shadow_rays;
        stats.batch_size = static_cast<int>(output.stats.batch_size);
        stats.batch_count = static_cast<int>(output.stats.batch_count);
        stats.status_counts = output.stats.status_counts;
        stats.cancelled = output.stats.cancelled ||
                          stats.completed_samples < stats.requested_samples;
        std::cout << "Rendering finished in " << stats.seconds
                  << " seconds (CUDA device " << stats.device_seconds
                  << " seconds)." << std::endl;
        RenderResult result;
        result.film = std::move(beauty);
        result.display = std::move(display);
        result.stats = std::move(stats);
        return result;
    }

  private:
    DeviceSceneStorage m_device_scene;
    PreparationStats m_preparation;
    std::string m_device_name;
};

std::unique_ptr<IRenderSession> make_cuda_render_session(const SceneIR &ir) {
    return std::make_unique<CudaRenderSession>(ir);
}

} // namespace cuda_backend

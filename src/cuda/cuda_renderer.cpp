#include "cuda_renderer.h"

#include "beauty_film.h"
#include "device_scene.h"
#include "scene_compiler.h"
#include "restir/restir_workspace.h"
#include "restir/restir_scheduler.h"
#include "wavefront_renderer.h"

#include <atomic>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>

namespace cuda_backend {

class CudaRenderSession final : public IRenderSession {
  public:
    explicit CudaRenderSession(const SceneIR &ir)
        : m_default_camera(ir.camera) {
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
        reset_history();
        if (integrator_descriptor(request.integrator).execution_model ==
            IntegratorExecutionModel::RestirFrame) {
            RenderFrameRequest frame;
            frame.render = request;
            frame.camera = m_default_camera;
            return render_restir_with_scene(
                m_device_scene.view(), frame, cancel, preview);
        }
        return render_with_scene(m_device_scene.view(), request, cancel,
                                 preview);
    }

    RenderResult render_frame(const RenderFrameRequest &request,
                              const CancellationToken &cancel,
                              PreviewSurface *preview) override {
        validate_render_frame_request(request);
        DeviceSceneView scene = m_device_scene.view();
        scene.scene.camera = compile_packed_camera(
            request.camera, scene.scene.scene_time0, scene.scene.scene_time1);
        if (integrator_descriptor(request.render.integrator).execution_model ==
            IntegratorExecutionModel::RestirFrame) {
            return render_restir_with_scene(scene, request, cancel, preview);
        }
        return render_with_scene(scene, request.render, cancel, preview);
    }

    void reset_history() override {
        m_restir_workspace.reset_history();
    }

  private:
    RenderResult render_with_scene(DeviceSceneView scene,
                                   const RenderRequest &request,
                                   const CancellationToken &cancel,
                                   PreviewSurface *preview) {
        bool expected = false;
        if (!m_rendering.compare_exchange_strong(expected, true)) {
            throw std::logic_error(
                "CUDA render session is already rendering");
        }
        struct Guard {
            std::atomic<bool> &state;
            ~Guard() {
                state.store(false, std::memory_order_relaxed);
            }
        } guard{m_rendering};
        validate_render_request(request);
        if (!integrator_supported(request.integrator, RenderBackend::CUDA)) {
            throw std::invalid_argument(
                "integrator is not supported by the CUDA backend");
        }
        const auto begin = std::chrono::steady_clock::now();
        CudaRenderSettings transport_settings;
        transport_settings.transport.policy =
            integrator_policy(request.integrator);
        transport_settings.transport.max_depth = request.max_depth;
        transport_settings.width = request.extent.width;
        transport_settings.height = request.extent.height;
        transport_settings.samples_per_pixel = request.samples_per_pixel;
        transport_settings.seed = request.seed;
        transport_settings.batch_size = request.cuda_batch_size;
        transport_settings.sample_clamp =
            static_cast<float>(request.sample_clamp);
        const CudaRenderOutput output = render_wavefront_cuda(
            scene, transport_settings, m_workspace, cancel.native_flag());

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
        stats.workspace_generation = output.stats.workspace_generation;
        stats.workspace_pixel_capacity =
            output.stats.workspace_pixel_capacity;
        stats.workspace_path_capacity =
            output.stats.workspace_path_capacity;
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

    RenderResult render_restir_with_scene(
        DeviceSceneView scene, const RenderFrameRequest &frame,
        const CancellationToken &cancel, PreviewSurface *preview) {
        bool expected = false;
        if (!m_rendering.compare_exchange_strong(expected, true)) {
            throw std::logic_error(
                "CUDA render session is already rendering");
        }
        struct Guard {
            std::atomic<bool> &state;
            ~Guard() {
                state.store(false, std::memory_order_relaxed);
            }
        } guard{m_rendering};
        validate_render_frame_request(frame);
        if (!integrator_supported(frame.render.integrator,
                                  RenderBackend::CUDA)) {
            throw std::invalid_argument(
                "integrator is not supported by the CUDA backend");
        }

        const auto begin = std::chrono::steady_clock::now();
        CudaRestirSkeletonSettings settings;
        settings.frame = frame;
        settings.reference_transport.policy =
            integrator_policy(IntegratorKind::MISPath);
        settings.reference_transport.max_depth = frame.render.max_depth;
        settings.generate_reference = false;
        const CudaRestirSchedulerOutput output =
            render_restir_skeleton_cuda(
                scene, settings, m_restir_workspace, cancel.native_flag());

        BeautyFilm beauty(frame.render.extent);
        const bool gi_mode =
            frame.render.integrator == IntegratorKind::ReSTIRGI;
        for (std::uint32_t y = 0; y < frame.render.extent.height; ++y) {
            for (std::uint32_t x = 0; x < frame.render.extent.width; ++x) {
                const std::uint32_t index =
                    y * frame.render.extent.width + x;
                const CudaFilmPixel &direct =
                    output.direct_film[y * frame.render.extent.width + x];
                Float3 radiance = direct.radiance;
                if (gi_mode) {
                    const CudaFilmPixel &indirect =
                        output.indirect_film[index];
                    if (indirect.sample_count != direct.sample_count) {
                        throw std::runtime_error(
                            "ReSTIR direct and indirect Film sample counts differ");
                    }
                    radiance = {
                        radiance.x + indirect.radiance.x,
                        radiance.y + indirect.radiance.y,
                        radiance.z + indirect.radiance.z,
                    };
                }
                beauty.set_pixel(
                    static_cast<int>(x), static_cast<int>(y),
                    color(radiance.x, radiance.y, radiance.z),
                    direct.sample_count);
            }
        }
        const auto resolve_begin = std::chrono::steady_clock::now();
        RenderBuffer display =
            resolve_beauty(beauty, frame.render.color_pipeline);
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
        stats.device_seconds = output.stats.milliseconds / 1000.0;
        stats.resolve_seconds =
            std::chrono::duration<double>(resolve_end - resolve_begin)
                .count();
        stats.width = static_cast<int>(frame.render.extent.width);
        stats.height = static_cast<int>(frame.render.extent.height);
        stats.samples_per_pixel =
            static_cast<int>(frame.render.samples_per_pixel);
        stats.requested_samples =
            static_cast<std::uint64_t>(frame.render.extent.pixel_count()) *
            frame.render.samples_per_pixel;
        stats.completed_samples = output.stats.sample_count;
        stats.sample_count = output.stats.sample_count;
        stats.seed = frame.render.seed;
        stats.clamped_samples = output.stats.di_clamped_samples;
        stats.invalid_samples = output.stats.di_invalid_samples +
                                output.stats.gi_invalid_samples;
        stats.backend = "cuda";
        stats.device_name = m_device_name;
        stats.scene_bytes = m_preparation.scene_bytes;
        stats.workspace_bytes = output.stats.workspace.bytes;
        stats.workspace_generation =
            output.stats.workspace.allocation_generation;
        stats.workspace_pixel_capacity =
            output.stats.workspace.pixel_capacity;
        stats.shadow_rays = output.stats.visibility_rays +
                            output.stats.gi_visibility_rays +
                            output.stats.gi_suffix_shadow_rays;
        stats.restir.iterations = output.stats.completed_iterations;
        stats.restir.initial_candidates = output.stats.initial_candidates;
        stats.restir.temporal_candidates = output.stats.temporal_candidates;
        stats.restir.temporal_accepted = output.stats.temporal_accepted;
        stats.restir.spatial_candidates = output.stats.spatial_candidates;
        stats.restir.spatial_accepted = output.stats.spatial_accepted;
        stats.restir.visibility_rays = output.stats.visibility_rays;
        stats.restir.history_resets =
            output.stats.history_reset_reason ==
                    restir::RestirHistoryResetReason::None
                ? 0u
                : 1u;
        stats.restir.invalid_reservoirs =
            output.stats.di_invalid_samples;
        stats.restir.shift_success = output.stats.temporal_accepted +
                                     output.stats.spatial_accepted;
        stats.restir.gi_initial_candidates =
            output.stats.gi_initial_candidates;
        stats.restir.gi_temporal_candidates =
            output.stats.gi_temporal_candidates;
        stats.restir.gi_temporal_accepted =
            output.stats.gi_temporal_accepted;
        stats.restir.gi_spatial_candidates =
            output.stats.gi_spatial_candidates;
        stats.restir.gi_spatial_accepted =
            output.stats.gi_spatial_accepted;
        stats.restir.gi_visibility_rays =
            output.stats.gi_visibility_rays;
        stats.restir.gi_fallbacks = output.stats.gi_fallbacks;
        stats.restir.gi_invalid_reservoirs =
            output.stats.gi_invalid_samples;
        stats.restir.average_M = output.stats.average_effective_M;
        stats.restir.average_age = output.stats.average_age;
        stats.restir.gi_average_M =
            output.stats.gi_average_effective_M;
        stats.restir.gi_average_age = output.stats.gi_average_age;
        for (std::size_t index = 0;
             index < stats.restir.shift_failures.size(); ++index) {
            stats.restir.shift_failures[index] =
                output.stats.gi_shift_failures[index];
        }
        for (std::size_t index = 0;
             index < stats.restir.history_failures.size(); ++index) {
            stats.restir.history_failures[index] =
                output.stats.temporal_rejection[index];
        }
        stats.cancelled = output.stats.cancelled ||
                          stats.completed_samples < stats.requested_samples;
        std::cout << "Rendering finished in " << stats.seconds
                  << " seconds (CUDA ReSTIR device "
                  << stats.device_seconds << " seconds)." << std::endl;
        RenderResult result;
        result.film = std::move(beauty);
        result.display = std::move(display);
        result.stats = std::move(stats);
        return result;
    }

    DeviceSceneStorage m_device_scene;
    CudaRenderWorkspace m_workspace;
    CudaRestirWorkspace m_restir_workspace;
    CameraConfig m_default_camera;
    PreparationStats m_preparation;
    std::string m_device_name;
    std::atomic<bool> m_rendering{false};
};

std::unique_ptr<IRenderSession> make_cuda_render_session(const SceneIR &ir) {
    return std::make_unique<CudaRenderSession>(ir);
}

} // namespace cuda_backend

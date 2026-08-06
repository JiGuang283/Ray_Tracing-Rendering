#include "cpu_render_session.h"

#include "camera.h"
#include "cpu_path_integrator.h"
#include "renderer.h"
#include "scene_loader.h"

#include <atomic>
#include <chrono>
#include <stdexcept>
#include <utility>

namespace {

class CpuRenderSession final : public IRenderSession {
  public:
    explicit CpuRenderSession(const SceneIR &ir) {
        const auto begin = std::chrono::steady_clock::now();
        m_scene = build_scene_config(ir);
        m_time0 = ir.time0;
        m_time1 = ir.time1;
        m_camera = make_camera(ir.camera);
        const auto end = std::chrono::steady_clock::now();
        m_preparation.compile_seconds =
            std::chrono::duration<double>(end - begin).count();
    }

    const PreparationStats &preparation_stats() const noexcept override {
        return m_preparation;
    }

    RenderResult render(const RenderRequest &request,
                        const CancellationToken &cancel,
                        PreviewSurface *preview) override {
        return render_with_camera(request, m_camera, cancel, preview);
    }

    RenderResult render_frame(const RenderFrameRequest &request,
                              const CancellationToken &cancel,
                              PreviewSurface *preview) override {
        validate_render_frame_request(request);
        return render_with_camera(request.render, make_camera(request.camera),
                                  cancel, preview);
    }

    void reset_history() override {
    }

  private:
    std::shared_ptr<camera> make_camera(const CameraConfig &config) const {
        validate_camera_config(config);
        return std::make_shared<camera>(
            config.lookfrom, config.lookat, config.vup, config.vfov,
            config.aspect_ratio, config.aperture, config.focus_dist, m_time0,
            m_time1);
    }

    RenderResult render_with_camera(const RenderRequest &request,
                                    const std::shared_ptr<camera> &camera,
                                    const CancellationToken &cancel,
                                    PreviewSurface *preview) {
        bool expected = false;
        if (!m_rendering.compare_exchange_strong(expected, true)) {
            throw std::logic_error("CPU render session is already rendering");
        }
        struct Guard {
            std::atomic<bool> &state;
            ~Guard() {
                state.store(false, std::memory_order_relaxed);
            }
        } guard{m_rendering};
        validate_render_request(request);
        if (!integrator_supported(request.integrator, RenderBackend::CPU)) {
            throw std::invalid_argument(
                "integrator is not supported by the CPU backend");
        }
        m_renderer.set_integrator(make_cpu_integrator(request.integrator));
        RenderResult result = m_renderer.render(
            m_scene.scene.world, camera, m_scene.preset.background,
            m_scene.scene.lights, request, cancel, preview);
        result.stats.compile_seconds = m_preparation.compile_seconds;
        result.stats.upload_seconds = m_preparation.upload_seconds;
        result.stats.scene_bytes = m_preparation.scene_bytes;
        return result;
    }

    SceneConfig m_scene;
    std::shared_ptr<camera> m_camera;
    double m_time0 = 0.0;
    double m_time1 = 1.0;
    Renderer m_renderer;
    PreparationStats m_preparation;
    std::atomic<bool> m_rendering{false};
};

} // namespace

std::unique_ptr<IRenderSession> make_cpu_render_session(const SceneIR &ir) {
    return std::make_unique<CpuRenderSession>(ir);
}

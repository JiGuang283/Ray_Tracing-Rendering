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
        m_camera = make_shared<camera>(
            ir.camera.lookfrom, ir.camera.lookat, ir.camera.vup,
            ir.camera.vfov, ir.camera.aspect_ratio, ir.camera.aperture,
            ir.camera.focus_dist, ir.time0, ir.time1);
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
            m_scene.scene.world, m_camera, m_scene.preset.background,
            m_scene.scene.lights, request, cancel, preview);
        result.stats.compile_seconds = m_preparation.compile_seconds;
        result.stats.upload_seconds = m_preparation.upload_seconds;
        result.stats.scene_bytes = m_preparation.scene_bytes;
        return result;
    }

  private:
    SceneConfig m_scene;
    std::shared_ptr<camera> m_camera;
    Renderer m_renderer;
    PreparationStats m_preparation;
    std::atomic<bool> m_rendering{false};
};

} // namespace

std::unique_ptr<IRenderSession> make_cpu_render_session(const SceneIR &ir) {
    return std::make_unique<CpuRenderSession>(ir);
}

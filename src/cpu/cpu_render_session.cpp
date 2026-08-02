#include "cpu_render_session.h"

#include "camera.h"
#include "direct_light_integrator.h"
#include "mis_path_integrator.h"
#include "path_integrator.h"
#include "pbr_path_integrator.h"
#include "renderer.h"
#include "rr_path_integrator.h"
#include "scene_loader.h"

#include <chrono>
#include <stdexcept>
#include <utility>

namespace {

std::shared_ptr<Integrator> make_integrator(IntegratorKind kind) {
    switch (kind) {
    case IntegratorKind::Path:
        return make_shared<PathIntegrator>();
    case IntegratorKind::RussianRoulette:
        return make_shared<RRPathInterator>();
    case IntegratorKind::PBRPath:
        return make_shared<PBRPathIntegrator>();
    case IntegratorKind::DirectLighting:
        return make_shared<DirectLightIntegrator>();
    case IntegratorKind::MISPath:
        return make_shared<MISPathIntegrator>();
    }
    throw std::invalid_argument("unsupported CPU integrator");
}

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
        m_renderer.set_integrator(make_integrator(request.integrator));
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
};

} // namespace

std::unique_ptr<IRenderSession> make_cpu_render_session(const SceneIR &ir) {
    return std::make_unique<CpuRenderSession>(ir);
}

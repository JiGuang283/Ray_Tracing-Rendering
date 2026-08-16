#ifndef RENDERER_H
#define RENDERER_H

#include "camera.h"
#include "hittable.h"
#include "integrator.h"
#include "preview_surface.h"
#include "render_result.h"
#include "render_types.h"
#include <atomic>
#include <memory>
#include <vector>

class LightSampler;

class Renderer {
  public:
    Renderer();

    void set_integrator(std::shared_ptr<Integrator> integrator) {
        m_integrator = integrator;
    }

    // light_sampler is an optional prebuilt scene-derived sampler. When
    // null the renderer builds one from lights, preserving the standalone
    // Renderer contract for tests and tools.
    RenderResult render(shared_ptr<hittable> world, shared_ptr<camera> cam,
                        const color &background,
                        const std::vector<shared_ptr<Light>> &lights,
                        const RenderRequest &request,
                        const CancellationToken &cancel = {},
                        PreviewSurface *preview = nullptr,
                        const LightSampler *light_sampler = nullptr);

    bool is_rendering() const {
        return m_is_rendering.load(std::memory_order_relaxed);
    }

  private:
    std::atomic<bool> m_is_rendering;

    std::shared_ptr<Integrator> m_integrator;
};

#endif

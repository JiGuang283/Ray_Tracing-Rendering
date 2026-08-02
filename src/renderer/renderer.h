#ifndef RENDERER_H
#define RENDERER_H

#include "camera.h"
#include "color_pipeline.h"
#include "hittable.h"
#include "integrator.h"
#include "material.h"
#include "preview_surface.h"
#include "render_buffer.h"
#include "render_result.h"
#include "render_types.h"
#include "rtweekend.h"
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

class Renderer {
  public:
    struct Settings {
        int samples_per_pixel = 10;
        unsigned seed = 1337;
        int thread_count = 0;
        double sample_clamp = 0.0;
        ColorPipelineSettings color_pipeline;
    };

    Renderer();

    void set_integrator(std::shared_ptr<Integrator> integrator) {
        m_integrator = integrator;
    }

    RenderStats render(shared_ptr<hittable> world, shared_ptr<camera> cam,
                       const color &background, RenderBuffer &target_buffer,
                       const std::vector<shared_ptr<Light>> &lights = {});

    RenderResult render(shared_ptr<hittable> world, shared_ptr<camera> cam,
                        const color &background,
                        const std::vector<shared_ptr<Light>> &lights,
                        const RenderRequest &request,
                        const CancellationToken &cancel = {},
                        PreviewSurface *preview = nullptr);

    void set_samples(int samples) {
        m_settings.samples_per_pixel = samples;
    }
    void set_seed(unsigned seed) {
        m_settings.seed = seed == 0 ? 1 : seed;
    }
    void set_thread_count(int thread_count) {
        m_settings.thread_count = thread_count;
    }
    void set_sample_clamp(double sample_clamp) {
        m_settings.sample_clamp = std::max(0.0, sample_clamp);
    }
    void set_color_pipeline(const ColorPipelineSettings &settings) {
        m_settings.color_pipeline = settings;
    }
    void set_max_depth(int depth) {
        if (m_integrator) {
            m_integrator->set_max_depth(depth);
        }
    }

    void cancel() {
        m_is_rendering = false;
    }
    bool is_rendering() const {
        return m_is_rendering;
    }

  private:
    Settings m_settings;
    std::atomic<bool> m_is_rendering;

    std::shared_ptr<Integrator> m_integrator;
};

#endif

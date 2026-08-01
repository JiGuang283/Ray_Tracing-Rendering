#include "test_harness.h"

#include "directional_light.h"
#include "hittable_list.h"
#include "integrator_common.h"
#include "light_sampler.h"
#include "quad_light.h"

#include <memory>
#include <vector>

namespace {

class TestLight final : public Light {
  public:
    TestLight(color power_value, color radiance, double sample_pdf,
              bool delta, bool bsdf_hittable)
        : m_power(power_value), m_radiance(radiance), m_pdf(sample_pdf),
          m_delta(delta), m_bsdf_hittable(bsdf_hittable) {
    }

    LightSample sample(const point3 &, const vec2 &) const override {
        LightSample result;
        result.Li = m_radiance;
        result.wi = vec3(0, 0, 1);
        result.pdf = m_pdf;
        result.dist = infinity;
        result.is_delta = m_delta;
        return result;
    }

    double pdf(const point3 &, const vec3 &) const override {
        return m_pdf;
    }

    bool is_delta() const override {
        return m_delta;
    }

    bool is_bsdf_hittable() const override {
        return m_bsdf_hittable;
    }

    color power() const override {
        return m_power;
    }

  private:
    color m_power;
    color m_radiance;
    double m_pdf = 0.0;
    bool m_delta = false;
    bool m_bsdf_hittable = false;
};

integrator_common::ShadedSurface make_diffuse_surface() {
    integrator_common::ShadedSurface surface;
    surface.surface.p = point3(0, 0, 0);
    surface.surface.geometry_normal = vec3(0, 0, 1);
    surface.surface.shading_normal = vec3(0, 0, 1);
    surface.surface.frame = ShadingFrame(vec3(0, 0, 1));
    surface.shading.reset(ShadingFrame(vec3(0, 0, 1)));
    surface.shading.bsdf.add_lambertian(color(1, 1, 1));
    surface.wo = vec3(0, 0, 1);
    return surface;
}

} // namespace

TEST_CASE(light_sampler_separates_delta_and_keeps_probability_floor) {
    auto strong = std::make_shared<TestLight>(
        color(1000, 1000, 1000), color(1, 1, 1), 1.0, false, true);
    auto dark = std::make_shared<TestLight>(
        color(0, 0, 0), color(0, 0, 0), 1.0, false, false);
    auto delta = std::make_shared<TestLight>(
        color(1e-9, 1e-9, 1e-9), color(1, 1, 1), 1.0, true, false);
    LightSampler sampler({strong, dark, delta});

    REQUIRE(sampler.delta_lights().size() == 1);
    REQUIRE(sampler.non_delta_light_count() == 2);
    REQUIRE_NEAR(sampler.non_delta_selection_pdf(0), 0.975, 1e-12);
    REQUIRE_NEAR(sampler.non_delta_selection_pdf(1), 0.025, 1e-12);
}

TEST_CASE(light_sampler_pdf_excludes_non_hittable_lights) {
    auto hittable = std::make_shared<TestLight>(
        color(1, 1, 1), color(1, 1, 1), 2.0, false, true);
    auto invisible = std::make_shared<TestLight>(
        color(1, 1, 1), color(1, 1, 1), 100.0, false, false);
    LightSampler sampler({hittable, invisible});

    REQUIRE_NEAR(sampler.pdf(point3(0, 0, 0), vec3(0, 0, 1)), 1.0,
                 1e-12);
}

TEST_CASE(delta_light_contribution_is_not_divided_by_light_selection) {
    auto dominant_but_dark = std::make_shared<TestLight>(
        color(1e9, 1e9, 1e9), color(0, 0, 0), 1.0, false, false);
    auto directional = std::make_shared<DirectionalLight>(
        vec3(0, 0, -1), color(0.1, 0.2, 0.3));
    LightSampler sampler({dominant_but_dark, directional});
    const auto surface = make_diffuse_surface();
    const hittable_list empty_scene;
    RNG rng(17);

    for (int i = 0; i < 32; ++i) {
        const color result = integrator_common::sample_direct_lighting(
            surface, empty_scene, sampler, rng, true);
        REQUIRE_NEAR(result.x(), 0.1 / pi, 1e-12);
        REQUIRE_NEAR(result.y(), 0.2 / pi, 1e-12);
        REQUIRE_NEAR(result.z(), 0.3 / pi, 1e-12);
    }
}

TEST_CASE(invisible_quad_has_no_bsdf_mis_competitor) {
    const QuadLight invisible(point3(-1, -1, 1), vec3(2, 0, 0),
                              vec3(0, 2, 0), color(1, 1, 1));
    const QuadLight geometry_backed(point3(-1, -1, 1), vec3(2, 0, 0),
                                    vec3(0, 2, 0), color(1, 1, 1), true);
    REQUIRE(!invisible.is_bsdf_hittable());
    REQUIRE(geometry_backed.is_bsdf_hittable());
}

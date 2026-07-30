#include "shading/bsdf.h"
#include "test_harness.h"

#include <cmath>
#include <stdexcept>

namespace {

color sample_weight(const BSDF &bsdf, const BSDFSample &sample) {
    const double cosine =
        sample.is_phase() ? 1.0 : bsdf.abs_cos_theta(sample.wi);
    return sample.f * cosine / sample.pdf;
}

bool finite_color(const color &value) {
    return std::isfinite(value.x()) && std::isfinite(value.y()) &&
           std::isfinite(value.z());
}

} // namespace

TEST_CASE(lambertian_eval_and_pdf_match_definition) {
    BSDF bsdf(ShadingFrame(vec3(0, 0, 1)));
    bsdf.add_lambertian(color(0.8, 0.4, 0.2));

    const vec3 wo(0, 0, 1);
    const vec3 wi = unit_vector(vec3(1, 0, 1));
    const color value = bsdf.eval(wo, wi);

    REQUIRE_NEAR(value.x(), 0.8 / pi, 1e-12);
    REQUIRE_NEAR(value.y(), 0.4 / pi, 1e-12);
    REQUIRE_NEAR(bsdf.pdf(wo, wi), wi.z() / pi, 1e-12);
}

TEST_CASE(delta_lobe_keeps_mixture_selection_probability) {
    BSDF bsdf(ShadingFrame(vec3(0, 0, 1)));
    bsdf.add_lambertian(color(0.5, 0.5, 0.5), 1.0);
    bsdf.add_specular_reflection(color(0.25, 0.25, 0.25), 1.0);

    RNG rng(17);
    int delta_count = 0;
    for (int i = 0; i < 20000; ++i) {
        auto sample = bsdf.sample(vec3(0, 0, 1), rng);
        REQUIRE(sample.has_value());
        REQUIRE(sample->pdf > 0.0);
        REQUIRE(finite_color(sample->f));
        if (sample->is_delta()) {
            ++delta_count;
            REQUIRE_NEAR(sample->pdf, 0.5, 1e-12);
            REQUIRE_NEAR(sample_weight(bsdf, *sample).x(), 0.5, 1e-12);
        }
    }
    REQUIRE(std::abs(delta_count / 20000.0 - 0.5) < 0.02);
}

TEST_CASE(dielectric_uses_fresnel_probability_and_eta) {
    BSDF bsdf(ShadingFrame(vec3(0, 0, 1)));
    bsdf.add_specular_dielectric(1.5, true);

    RNG rng(41);
    int reflected = 0;
    int transmitted = 0;
    for (int i = 0; i < 100000; ++i) {
        auto sample = bsdf.sample(vec3(0, 0, 1), rng);
        REQUIRE(sample.has_value());
        if (sample->is_transmission()) {
            ++transmitted;
            REQUIRE_NEAR(sample->eta, 1.5, 1e-12);
            REQUIRE_NEAR(sample_weight(bsdf, *sample).x(), 1.0 / 2.25,
                         1e-12);
        } else {
            ++reflected;
            REQUIRE_NEAR(sample->eta, 1.0, 1e-12);
            REQUIRE_NEAR(sample_weight(bsdf, *sample).x(), 1.0, 1e-12);
        }
    }
    REQUIRE(transmitted > 0);
    REQUIRE(std::abs(reflected / 100000.0 - 0.04) < 0.004);
}

TEST_CASE(bsdf_capacity_overflow_is_not_silent) {
    BSDF bsdf(ShadingFrame(vec3(0, 0, 1)));
    for (std::size_t i = 0; i < BSDF::kMaxClosures; ++i) {
        bsdf.add_lambertian(color(0.5, 0.5, 0.5));
    }

    bool threw = false;
    try {
        bsdf.add_lambertian(color(0.5, 0.5, 0.5));
    } catch (const std::overflow_error &) {
        threw = true;
    }
    REQUIRE(threw);
}

TEST_CASE(ggx_samples_are_finite_and_nonnegative) {
    BSDF bsdf(ShadingFrame(vec3(0, 0, 1)));
    bsdf.add_microfacet_ggx(color(0.8, 0.6, 0.2), 0.35, 0.8);
    RNG rng(99);

    for (int i = 0; i < 20000; ++i) {
        auto sample = bsdf.sample(unit_vector(vec3(0.7, 0, 1)), rng);
        if (!sample) {
            continue;
        }
        REQUIRE(sample->pdf > 0.0);
        REQUIRE(finite_color(sample->f));
        REQUIRE(sample->f.x() >= 0.0);
        REQUIRE(sample->f.y() >= 0.0);
        REQUIRE(sample->f.z() >= 0.0);
    }
}

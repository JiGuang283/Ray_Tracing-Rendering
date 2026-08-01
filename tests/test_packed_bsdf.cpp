#include "test_harness.h"

#include "render_data/packed_bsdf.h"

#include <cmath>
#include <cstdint>

namespace {

PackedMaterialOutput make_output() {
    PackedMaterialOutput output{};
    output.frame.tangent = {1.0f, 0.0f, 0.0f};
    output.frame.normal = {0.0f, 0.0f, 1.0f};
    output.frame.handedness = 1.0f;
    output.geometry_normal = {0.0f, 0.0f, 1.0f};
    return output;
}

void add_closure(PackedMaterialOutput &output, PackedClosureType type,
                 Float4 parameters, float contribution_weight = 1.0f,
                 float sample_weight = 1.0f,
                 std::uint32_t flags = PACKED_CLOSURE_NONE) {
    REQUIRE(output.closure_count < PackedMaterialOutput::kMaxClosures);
    PackedClosure &closure = output.closures[output.closure_count++];
    closure.type = type;
    closure.parameters = parameters;
    closure.contribution_weight = contribution_weight;
    closure.sample_weight = sample_weight;
    closure.flags = flags;
}

bool finite(Float3 value) {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z);
}

} // namespace

TEST_CASE(packed_bsdf_lambertian_eval_pdf_and_sample) {
    PackedMaterialOutput output = make_output();
    add_closure(output, PackedClosureType::Lambertian,
                {0.8f, 0.4f, 0.2f, 0.0f});

    Float3 value{};
    float pdf = 0.0f;
    REQUIRE(evaluate_packed_bsdf(output, {0, 0, 1}, {0, 0, 1}, value) ==
            PackedBSDFStatus::Success);
    REQUIRE(packed_bsdf_pdf(output, {0, 0, 1}, {0, 0, 1}, pdf) ==
            PackedBSDFStatus::Success);
    REQUIRE_NEAR(value.x, 0.8 / 3.14159265358979323846, 1e-6);
    REQUIRE_NEAR(value.y, 0.4 / 3.14159265358979323846, 1e-6);
    REQUIRE_NEAR(value.z, 0.2 / 3.14159265358979323846, 1e-6);
    REQUIRE_NEAR(pdf, 1.0 / 3.14159265358979323846, 1e-6);

    RNG rng(17);
    for (int index = 0; index < 256; ++index) {
        PackedBSDFSample sample{};
        REQUIRE(sample_packed_bsdf(output, {0, 0, 1}, rng, sample) ==
                PackedBSDFStatus::Success);
        REQUIRE(sample.wi.z > 0.0f);
        REQUIRE(!sample.is_delta());
        REQUIRE((sample.flags & PACKED_BSDF_DIFFUSE) != 0);
        REQUIRE(sample.pdf > 0.0f);
        REQUIRE(finite(sample.f));
    }
}

TEST_CASE(packed_bsdf_geometry_normal_blocks_backside_reflection) {
    PackedMaterialOutput output = make_output();
    output.frame.normal = {0.8f, 0.0f, 0.6f};
    add_closure(output, PackedClosureType::Lambertian,
                {1.0f, 1.0f, 1.0f, 0.0f});

    Float3 value{1, 1, 1};
    float pdf = 1.0f;
    REQUIRE(evaluate_packed_bsdf(output, {0, 0, 1}, {0, 0, -1}, value) ==
            PackedBSDFStatus::Success);
    REQUIRE(packed_bsdf_pdf(output, {0, 0, 1}, {0, 0, -1}, pdf) ==
            PackedBSDFStatus::Success);
    REQUIRE_NEAR(value.x, 0.0, 0.0);
    REQUIRE_NEAR(value.y, 0.0, 0.0);
    REQUIRE_NEAR(value.z, 0.0, 0.0);
    REQUIRE_NEAR(pdf, 0.0, 0.0);
}

TEST_CASE(packed_bsdf_delta_selection_pdf_is_discrete_probability) {
    PackedMaterialOutput output = make_output();
    add_closure(output, PackedClosureType::Lambertian,
                {0.5f, 0.5f, 0.5f, 0.0f}, 1.0f, 1.0f);
    add_closure(output, PackedClosureType::Mirror,
                {0.8f, 0.6f, 0.4f, 0.0f}, 0.5f, 3.0f);

    RNG rng(1234);
    int mirror_count = 0;
    constexpr int kSamples = 20000;
    for (int index = 0; index < kSamples; ++index) {
        PackedBSDFSample sample{};
        const PackedBSDFStatus status =
            sample_packed_bsdf(output, {0, 0, 1}, rng, sample);
        REQUIRE(status == PackedBSDFStatus::Success);
        if (sample.is_delta()) {
            ++mirror_count;
            REQUIRE(sample.closure_index == 1);
            REQUIRE_NEAR(sample.pdf, 0.75, 1e-6);
            REQUIRE_NEAR(sample.f.x, 0.4, 1e-6);
            REQUIRE_NEAR(sample.f.y, 0.3, 1e-6);
            REQUIRE_NEAR(sample.f.z, 0.2, 1e-6);
        }
    }
    const double frequency = static_cast<double>(mirror_count) / kSamples;
    REQUIRE_NEAR(frequency, 0.75, 0.02);
}

TEST_CASE(packed_bsdf_dielectric_uses_exact_fresnel_and_eta) {
    PackedMaterialOutput output = make_output();
    add_closure(output, PackedClosureType::Dielectric,
                {1.5f, 0.0f, 0.0f, 0.0f}, 1.0f, 1.0f,
                PACKED_CLOSURE_FRONT_FACE);

    RNG rng(9182);
    int reflected = 0;
    int transmitted = 0;
    constexpr int kSamples = 30000;
    for (int index = 0; index < kSamples; ++index) {
        PackedBSDFSample sample{};
        REQUIRE(sample_packed_bsdf(output, {0, 0, 1}, rng, sample) ==
                PackedBSDFStatus::Success);
        REQUIRE(sample.is_delta());
        if (sample.is_transmission()) {
            ++transmitted;
            REQUIRE_NEAR(sample.eta, 1.5, 1e-6);
            REQUIRE(sample.wi.z < 0.0f);
            const float throughput = sample.f.x * std::abs(sample.wi.z) /
                                     sample.pdf;
            REQUIRE_NEAR(throughput, 1.0 / (1.5 * 1.5), 1e-5);
        } else {
            ++reflected;
            REQUIRE(sample.wi.z > 0.0f);
        }
    }
    REQUIRE(transmitted + reflected == kSamples);
    REQUIRE_NEAR(static_cast<double>(reflected) / kSamples, 0.04, 0.008);
}

TEST_CASE(packed_bsdf_ggx_vndf_stays_finite_at_grazing_angles) {
    PackedMaterialOutput output = make_output();
    add_closure(output, PackedClosureType::GGXReflection,
                {0.7f, 0.4f, 0.2f, 0.35f});

    RNG rng(7123);
    int successes = 0;
    const Float3 wo{0.99995f, 0.0f, 0.01f};
    for (int index = 0; index < 10000; ++index) {
        PackedBSDFSample sample{};
        const PackedBSDFStatus status =
            sample_packed_bsdf(output, wo, rng, sample);
        REQUIRE(status == PackedBSDFStatus::Success ||
                status == PackedBSDFStatus::NoSample);
        if (status == PackedBSDFStatus::Success) {
            ++successes;
            REQUIRE(sample.pdf > 0.0f);
            REQUIRE(std::isfinite(sample.pdf));
            REQUIRE(finite(sample.f));
            REQUIRE(finite(sample.wi));
            REQUIRE(sample.wi.z > 0.0f);
        }
    }
    REQUIRE(successes > 9000);
}

TEST_CASE(packed_bsdf_isotropic_phase_has_unit_transport_cosine) {
    PackedMaterialOutput output = make_output();
    add_closure(output, PackedClosureType::IsotropicPhase,
                {0.6f, 0.7f, 0.8f, 0.0f});
    REQUIRE(packed_bsdf_is_phase(output));
    REQUIRE_NEAR(packed_bsdf_abs_cos_theta(output, {1, 0, 0}), 1.0, 0.0);

    Float3 value{};
    float pdf = 0.0f;
    REQUIRE(evaluate_packed_bsdf(output, {0, 0, 1}, {0, 0, -1}, value) ==
            PackedBSDFStatus::Success);
    REQUIRE(packed_bsdf_pdf(output, {0, 0, 1}, {0, 0, -1}, pdf) ==
            PackedBSDFStatus::Success);
    REQUIRE_NEAR(value.x, 0.6 / (4.0 * 3.14159265358979323846), 1e-6);
    REQUIRE_NEAR(pdf, 1.0 / (4.0 * 3.14159265358979323846), 1e-6);
}

TEST_CASE(packed_bsdf_rejects_empty_and_invalid_inputs) {
    PackedMaterialOutput output = make_output();
    RNG rng(1);
    PackedBSDFSample sample{};
    REQUIRE(sample_packed_bsdf(output, {0, 0, 1}, rng, sample) ==
            PackedBSDFStatus::Empty);

    add_closure(output, PackedClosureType::Lambertian,
                {1.0f, 1.0f, 1.0f, 0.0f});
    REQUIRE(sample_packed_bsdf(output, {0, 0, 0}, rng, sample) ==
            PackedBSDFStatus::InvalidInput);
    output.closure_count = PackedMaterialOutput::kMaxClosures + 1;
    REQUIRE(sample_packed_bsdf(output, {0, 0, 1}, rng, sample) ==
            PackedBSDFStatus::InvalidInput);
}

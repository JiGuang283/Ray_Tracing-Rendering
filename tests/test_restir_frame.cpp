#include "test_harness.h"

#include "restir_history.h"
#include "restir_di_core.h"
#include "restir_spatial_core.h"
#include "restir_spatial_pairwise_core.h"
#include "restir_surface.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <string>

namespace {

float dot(Float3 left, Float3 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

Float3 normalized(Float3 value) {
    const float inverse =
        1.0f / std::sqrt(value.x * value.x + value.y * value.y +
                         value.z * value.z);
    return {value.x * inverse, value.y * inverse, value.z * inverse};
}

RenderFrameRequest frame_request(std::uint64_t frame_index = 0) {
    RenderFrameRequest request;
    request.render.extent = make_image_extent(32, 18);
    request.render.integrator = IntegratorKind::ReSTIRDI;
    request.render.samples_per_pixel = 1;
    request.render.max_depth = 4;
    request.camera.lookfrom = point3(0, 1, 4);
    request.camera.lookat = point3(0, 1, 0);
    request.camera.vup = vec3(0, 1, 0);
    request.camera.aspect_ratio = 16.0 / 9.0;
    request.frame_index = frame_index;
    return request;
}

} // namespace

TEST_CASE(restir_surface_has_stable_layout_and_hit_reconstruction) {
    REQUIRE(sizeof(restir::RestirSurface) == 64u);
    REQUIRE(alignof(restir::RestirSurface) == 16u);
    restir::RestirSurface surface;
    surface.hit_t = 3.5f;
    surface.barycentric_u = 0.2f;
    surface.barycentric_v = 0.3f;
    surface.instance_id = 7u;
    surface.primitive_id = 11u;
    surface.flags = PACKED_HIT_TRIANGLE | PACKED_HIT_FRONT_FACE |
                    restir::RESTIR_SURFACE_VALID |
                    restir::RESTIR_SURFACE_DELTA_ONLY;
    const PackedHit hit = restir::restir_surface_hit(surface);
    REQUIRE_NEAR(hit.t, 3.5f, 1e-6);
    REQUIRE_NEAR(hit.barycentric_u, 0.2f, 1e-6);
    REQUIRE_NEAR(hit.barycentric_v, 0.3f, 1e-6);
    REQUIRE(hit.instance_id == 7u);
    REQUIRE(hit.primitive_id == 11u);
    REQUIRE(hit.flags == (PACKED_HIT_TRIANGLE | PACKED_HIT_FRONT_FACE));
    REQUIRE(surface.valid());
    REQUIRE(surface.delta_only());
}

TEST_CASE(restir_di_sample_and_reservoir_abi_is_stable) {
    REQUIRE(sizeof(restir::RestirLightSample) == 32u);
    REQUIRE(sizeof(restir::RestirDIReservoir) == 64u);
    restir::RestirDIReservoir reservoir;
    restir::reset_reservoir(reservoir);
    REQUIRE(!restir::reservoir_has_sample(reservoir));
    REQUIRE(reservoir.M == 0u);
    REQUIRE_NEAR(reservoir.effective_M, 0.0f, 0.0f);
}

TEST_CASE(restir_di_reservoir_tracks_literal_and_effective_candidate_mass) {
    restir::RestirDIReservoir reservoir;
    restir::reset_reservoir(reservoir);
    restir::RestirLightSample sample;
    sample.light_id = 3u;

    const auto first = restir::stream_di_weight(
        reservoir, sample, 2.0f, 6.0f, 4u, 1.5f, 0.25f);
    REQUIRE(first.accepted());
    REQUIRE(first.changed_selection());
    REQUIRE(reservoir.M == 4u);
    REQUIRE_NEAR(reservoir.effective_M, 1.5f, 1e-6f);

    const auto finalized = restir::finalize_reservoir(reservoir);
    REQUIRE(finalized.accepted());
    REQUIRE(restir::reservoir_is_usable(reservoir));
    REQUIRE_NEAR(reservoir.unbiased_contribution_weight, 2.0f, 1e-6f);
}

TEST_CASE(restir_di_initial_candidates_keep_effective_mass_equal_to_count) {
    restir::RestirDIReservoir reservoir;
    restir::reset_reservoir(reservoir);
    restir::RestirLightSample sample;
    sample.light_id = 5u;
    const auto candidate = restir::make_candidate(
        sample, 3.0f, 0.5f, true, true);

    REQUIRE(restir::stream_candidate(reservoir, candidate, 0.1f).accepted());
    REQUIRE(restir::stream_candidate(reservoir, candidate, 0.9f).accepted());
    REQUIRE(reservoir.M == 2u);
    REQUIRE_NEAR(reservoir.effective_M, 2.0f, 1e-6f);
    REQUIRE(restir::finalize_reservoir(reservoir).accepted());
    REQUIRE_NEAR(reservoir.unbiased_contribution_weight, 2.0f, 1e-6f);
}

TEST_CASE(restir_spatial_neighbors_are_deterministic_unique_and_not_self) {
    constexpr std::uint32_t width = 64u;
    constexpr std::uint32_t height = 64u;
    constexpr std::uint32_t center = 32u * width + 32u;
    std::array<std::uint32_t, 64> pixels{};
    for (std::uint32_t index = 0; index < pixels.size(); ++index) {
        const auto first = restir::restir_spatial_neighbor(
            center, index, width, height, 7u, 2u, 123u);
        const auto second = restir::restir_spatial_neighbor(
            center, index, width, height, 7u, 2u, 123u);
        REQUIRE(first.valid == 1u);
        REQUIRE(first.pixel == second.pixel);
        REQUIRE(first.pixel != center);
        pixels[index] = first.pixel;
        for (std::uint32_t previous = 0; previous < index; ++previous) {
            REQUIRE(pixels[previous] != pixels[index]);
        }
    }
    const auto outside = restir::restir_spatial_neighbor(
        0u, 0u, width, height, 7u, 2u, 123u);
    REQUIRE(outside.valid == 0u || outside.pixel != 0u);
}

TEST_CASE(restir_spatial_compatibility_rejects_surface_discontinuities) {
    restir::RestirSurface center;
    center.flags = restir::RESTIR_SURFACE_VALID;
    center.material_id = 2u;
    center.view_depth = 10.0f;
    center.shading_normal =
        restir::pack_octahedral_normal({0.0f, 1.0f, 0.0f});
    restir::RestirSurface neighbor = center;
    neighbor.view_depth = 10.5f;
    REQUIRE(restir::restir_spatial_compatibility(center, neighbor, 0.9f,
                                                  0.1f) ==
            restir::RestirSpatialCompatibility::Compatible);

    neighbor.material_id = 3u;
    REQUIRE(restir::restir_spatial_compatibility(center, neighbor, 0.9f,
                                                  0.1f) ==
            restir::RestirSpatialCompatibility::MaterialMismatch);
    neighbor = center;
    neighbor.view_depth = 12.0f;
    REQUIRE(restir::restir_spatial_compatibility(center, neighbor, 0.9f,
                                                  0.1f) ==
            restir::RestirSpatialCompatibility::DepthMismatch);
    neighbor = center;
    neighbor.shading_normal =
        restir::pack_octahedral_normal({0.0f, -1.0f, 0.0f});
    REQUIRE(restir::restir_spatial_compatibility(center, neighbor, 0.9f,
                                                  0.1f) ==
            restir::RestirSpatialCompatibility::NormalMismatch);
}

TEST_CASE(restir_pairwise_weights_and_effective_mass_are_explicit) {
    REQUIRE_NEAR(restir::restir_pairwise_mis_weight(2.0f, 1.0f, 3.0f,
                                                    4.0f),
                 0.6f, 1e-6f);
    REQUIRE_NEAR(restir::restir_pairwise_m_factor(0.0f, 0.0f), 1.0f,
                 1e-6f);
    REQUIRE_NEAR(restir::restir_pairwise_m_factor(2.0f, 1.0f),
                 1.0f / 256.0f, 1e-6f);

    restir::RestirDIReservoir reservoir;
    restir::reset_reservoir(reservoir);
    restir::RestirLightSample sample;
    sample.light_id = 1u;
    REQUIRE(restir::stream_di_weight(reservoir, sample, 2.0f, 8.0f,
                                     4u, 0.5f, 0.0f)
                .accepted());
    REQUIRE(restir::finalize_di_reservoir(reservoir, 2.0f).accepted());
    REQUIRE_NEAR(reservoir.effective_M, 0.5f, 1e-6f);
    REQUIRE_NEAR(reservoir.unbiased_contribution_weight, 2.0f, 1e-6f);
}

TEST_CASE(restir_octahedral_normals_round_trip_both_hemispheres) {
    const std::array<Float3, 8> normals{{
        {1.0f, 0.0f, 0.0f},
        {-1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, -1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f},
        {0.0f, 0.0f, -1.0f},
        normalized({0.3f, -0.7f, 0.4f}),
        normalized({-0.6f, 0.2f, -0.9f}),
    }};
    for (const Float3 normal : normals) {
        const Float3 decoded = restir::unpack_octahedral_normal(
            restir::pack_octahedral_normal(normal));
        REQUIRE(dot(normal, decoded) >= 0.9999f);
    }
}

TEST_CASE(restir_history_only_advances_after_iteration_commit) {
    restir::RestirFrameState state;
    RenderFrameRequest request = frame_request(10u);
    const restir::RestirFramePreparation first =
        restir::prepare_restir_frame(state, request);
    REQUIRE(first.reset_reason ==
            restir::RestirHistoryResetReason::Uninitialized);
    REQUIRE(first.read_gbuffer == restir::kInvalidHistoryBuffer);
    REQUIRE(first.read_di_reservoir == restir::kInvalidHistoryBuffer);
    REQUIRE(first.write_gbuffer == 0u);
    REQUIRE(first.write_di_reservoir == 0u);
    REQUIRE(state.history_valid == 0u);

    const restir::RestirFramePreparation cancelled =
        restir::prepare_restir_frame(state, request);
    REQUIRE(cancelled.reset_reason ==
            restir::RestirHistoryResetReason::Uninitialized);
    REQUIRE(cancelled.write_gbuffer == 0u);
    REQUIRE(cancelled.write_di_reservoir == 0u);
    REQUIRE(state.completed_iterations == 0u);

    restir::commit_restir_iteration(
        state, first.write_gbuffer, first.write_di_reservoir,
        request.frame_index);
    REQUIRE(state.history_valid == 1u);
    REQUIRE(state.committed_gbuffer == 0u);
    REQUIRE(state.committed_di_reservoir == 0u);
    REQUIRE(state.completed_iterations == 1u);

    request.frame_index = 11u;
    const restir::RestirFramePreparation second =
        restir::prepare_restir_frame(state, request);
    REQUIRE(!second.reset());
    REQUIRE(second.read_gbuffer == 0u);
    REQUIRE(second.read_di_reservoir == 0u);
    REQUIRE(second.write_gbuffer == 1u);
    REQUIRE(second.write_di_reservoir == 1u);
    restir::commit_restir_iteration(
        state, second.write_gbuffer, second.write_di_reservoir,
        request.frame_index);
    REQUIRE(state.committed_gbuffer == 1u);
    REQUIRE(state.committed_di_reservoir == 1u);
    REQUIRE(state.completed_iterations == 2u);

    request.frame_index = 12u;
    const restir::RestirFramePreparation third =
        restir::prepare_restir_frame(state, request);
    REQUIRE(third.read_di_reservoir == 1u);
    REQUIRE(third.write_di_reservoir == 2u);
    restir::commit_restir_iteration(
        state, third.write_gbuffer, third.write_di_reservoir,
        request.frame_index);
    REQUIRE(state.committed_di_reservoir == 2u);
}

TEST_CASE(restir_history_tracks_gbuffer_and_di_reservoir_independently) {
    restir::RestirFrameState state;
    RenderFrameRequest request = frame_request(4u);
    const restir::RestirFramePreparation initial =
        restir::prepare_restir_frame(state, request);
    restir::commit_restir_iteration(state, initial.write_gbuffer, 1u,
                                    request.frame_index);
    REQUIRE(state.committed_gbuffer == 0u);
    REQUIRE(state.committed_di_reservoir == 1u);

    request.frame_index = 5u;
    const restir::RestirFramePreparation next =
        restir::prepare_restir_frame(state, request);
    REQUIRE(next.read_gbuffer == 0u);
    REQUIRE(next.write_gbuffer == 1u);
    REQUIRE(next.read_di_reservoir == 1u);
    REQUIRE(next.write_di_reservoir == 2u);
}

TEST_CASE(restir_history_classifies_incompatible_keys) {
    restir::RestirFrameState state;
    RenderFrameRequest request = frame_request(3u);
    const auto initial = restir::prepare_restir_frame(state, request);
    restir::commit_restir_iteration(
        state, initial.write_gbuffer, initial.write_di_reservoir,
        request.frame_index);
    const std::uint64_t generation = state.history_generation;

    request.render.restir.spatial_neighbors += 1u;
    const auto settings_reset = restir::prepare_restir_frame(state, request);
    REQUIRE(settings_reset.reset_reason ==
            restir::RestirHistoryResetReason::SettingsChanged);
    REQUIRE(state.history_generation == generation + 1u);
    REQUIRE(state.history_valid == 0u);
    restir::commit_restir_iteration(
        state, settings_reset.write_gbuffer,
        settings_reset.write_di_reservoir, request.frame_index);

    request.revision.camera += 1u;
    const auto camera_reset = restir::prepare_restir_frame(state, request);
    REQUIRE(camera_reset.reset_reason ==
            restir::RestirHistoryResetReason::CameraRevisionChanged);
    restir::commit_restir_iteration(
        state, camera_reset.write_gbuffer,
        camera_reset.write_di_reservoir, request.frame_index);

    request.frame_index = 2u;
    const auto rewind = restir::prepare_restir_frame(state, request);
    REQUIRE(rewind.reset_reason ==
            restir::RestirHistoryResetReason::FrameIndexRewound);
}

TEST_CASE(restir_explicit_history_reset_clears_committed_state) {
    restir::RestirFrameState state;
    RenderFrameRequest request = frame_request();
    auto preparation = restir::prepare_restir_frame(state, request);
    restir::commit_restir_iteration(
        state, preparation.write_gbuffer,
        preparation.write_di_reservoir, 0u);
    const std::uint64_t generation = state.history_generation;

    restir::reset_restir_history(state);
    REQUIRE(state.history_valid == 0u);
    REQUIRE(state.completed_iterations == 0u);
    REQUIRE(state.history_generation == generation + 1u);

    request.render.restir.history_mode = RestirHistoryMode::Reset;
    preparation = restir::prepare_restir_frame(state, request);
    REQUIRE(preparation.reset_reason ==
            restir::RestirHistoryResetReason::Explicit);
    REQUIRE(std::string(restir::restir_history_reset_reason_name(
                preparation.reset_reason)) == "explicit");
}

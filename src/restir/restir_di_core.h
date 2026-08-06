#ifndef RESTIR_DI_CORE_H
#define RESTIR_DI_CORE_H

#include "packed_bsdf_core.h"
#include "packed_transport_core.h"
#include "restir_di_reservoir_core.h"
#include "restir_gbuffer_core.h"
#include "restir_light_core.h"

namespace restir {

enum class RestirDIStatus : std::uint32_t {
    Success = 0,
    NoSurface = 1,
    UnsupportedSurface = 2,
    ReconstructionFailure = 3,
    MaterialFailure = 4,
    LightFailure = 5,
    BSDFFailure = 6,
    ReservoirEmpty = 7,
    ReservoirFailure = 8,
    TraversalFailure = 9,
    NonFinite = 10,
};

struct RestirDIPixelContext {
    PackedRay camera_ray{};
    PackedSurfaceInteraction surface{};
    PackedMaterialOutput material{};
    Float3 wo{};
};

struct RestirDICandidateStats {
    std::uint32_t attempted = 0;
    std::uint32_t represented = 0;
    std::uint32_t rejected = 0;
};

RT_HOST_DEVICE RT_FORCE_INLINE std::uint32_t restir_random_seed(
    std::uint32_t seed, std::uint32_t pixel, std::uint32_t iteration,
    std::uint32_t domain) noexcept {
    return mix_seed(mix_seed(mix_seed(seed, pixel), iteration), domain);
}

RT_HOST_DEVICE RT_FORCE_INLINE RestirDIStatus reconstruct_restir_di_context(
    const CompiledSceneView &scene, const RestirSurface &stored,
    const PackedCamera &camera, std::uint32_t width, std::uint32_t height,
    std::uint32_t camera_pixel, std::uint32_t iteration,
    std::uint32_t seed,
    RestirDIPixelContext &context) noexcept {
    context = {};
    if (!stored.valid()) {
        return RestirDIStatus::NoSurface;
    }
    if ((stored.flags & RESTIR_SURFACE_UNSUPPORTED_DOMAIN) != 0u) {
        return RestirDIStatus::UnsupportedSurface;
    }
    if (width < 2u || height < 2u ||
        camera_pixel >= static_cast<std::uint64_t>(width) * height) {
        return RestirDIStatus::ReconstructionFailure;
    }
    RNG camera_rng(packed_transport::packed_camera_sample_seed(
        seed, camera_pixel, iteration));
    context.camera_ray = packed_transport::generate_packed_camera_ray_core(
        camera, camera_pixel % width, camera_pixel / width, width, height,
        camera_rng);
    const PackedHit hit = restir_surface_hit(stored);
    const PackedShadingStatus reconstruction =
        packed_reconstruction::reconstruct_compiled_hit_core(
            scene, context.camera_ray, hit, context.surface);
    if (reconstruction != PackedShadingStatus::Success) {
        return reconstruction == PackedShadingStatus::NonFinite
                   ? RestirDIStatus::NonFinite
                   : RestirDIStatus::ReconstructionFailure;
    }
    const PackedShadingStatus material_status =
        packed_material::evaluate_packed_material_core(
            scene, context.surface.material_id, context.surface,
            context.material);
    if (material_status != PackedShadingStatus::Success) {
        return material_status == PackedShadingStatus::NonFinite
                   ? RestirDIStatus::NonFinite
                   : RestirDIStatus::MaterialFailure;
    }
    context.wo = packed_transport::math::normalize(
        packed_transport::math::multiply(context.camera_ray.direction,
                                         -1.0f));
    return packed_transport::math::finite(context.wo)
               ? RestirDIStatus::Success
               : RestirDIStatus::NonFinite;
}

RT_HOST_DEVICE RT_FORCE_INLINE RestirDIStatus reconstruct_restir_di_context(
    const CompiledSceneView &scene, const RestirSurface &stored,
    std::uint32_t width, std::uint32_t height, std::uint32_t pixel,
    std::uint32_t iteration, std::uint32_t seed,
    RestirDIPixelContext &context) noexcept {
    return reconstruct_restir_di_context(
        scene, stored, scene.camera, width, height, pixel, iteration, seed,
        context);
}

RT_HOST_DEVICE RT_FORCE_INLINE RestirDIStatus
evaluate_unshadowed_restir_direct(
    const RestirDIPixelContext &context,
    const PackedLightSample &light_sample, Float3 &integrand,
    float &target) noexcept {
    integrand = {};
    target = 0.0f;
    if (!(light_sample.pdf > 0.0f) ||
        !packed_transport::math::finite(light_sample.pdf) ||
        !packed_transport::math::finite(light_sample.wi) ||
        !packed_transport::math::finite(light_sample.radiance)) {
        return RestirDIStatus::LightFailure;
    }
    Float3 f{};
    const PackedBSDFStatus bsdf_status =
        packed_bsdf::eval_packed_bsdf_core(
            context.material, context.wo, light_sample.wi, f);
    if (bsdf_status == PackedBSDFStatus::Empty) {
        return RestirDIStatus::Success;
    }
    if (bsdf_status != PackedBSDFStatus::Success) {
        return RestirDIStatus::BSDFFailure;
    }
    const float cosine = packed_bsdf::abs_cos_theta(
        context.material, light_sample.wi);
    if (!(cosine > 0.0f) || !packed_transport::math::finite(cosine) ||
        packed_transport::math::black(f) ||
        packed_transport::math::black(light_sample.radiance)) {
        return RestirDIStatus::Success;
    }
    integrand = packed_transport::math::multiply(
        packed_transport::math::multiply(f, light_sample.radiance),
        cosine);
    target = packed_light::math::luminance(integrand);
    if (!packed_transport::math::finite(integrand) ||
        !packed_transport::math::finite(target)) {
        integrand = {};
        target = 0.0f;
        return RestirDIStatus::NonFinite;
    }
    return RestirDIStatus::Success;
}

RT_HOST_DEVICE RT_FORCE_INLINE RestirDIStatus generate_initial_di_reservoir(
    const CompiledSceneView &scene, const RestirSurface &surface,
    std::uint32_t width, std::uint32_t height, std::uint32_t pixel,
    std::uint32_t iteration, std::uint32_t seed,
    std::uint32_t candidate_count, RestirDIReservoir &reservoir,
    RestirDICandidateStats &stats) noexcept {
    reset_reservoir(reservoir);
    stats = {};
    if (candidate_count == 0u) {
        return RestirDIStatus::ReservoirEmpty;
    }
    RestirDIPixelContext context;
    const RestirDIStatus context_status = reconstruct_restir_di_context(
        scene, surface, width, height, pixel, iteration, seed, context);
    if (context_status != RestirDIStatus::Success) {
        return context_status;
    }
    if (context.material.closure_count == 0u || surface.delta_only()) {
        return RestirDIStatus::UnsupportedSurface;
    }
    if (scene.non_delta_light_indices.count == 0u) {
        return RestirDIStatus::ReservoirEmpty;
    }

    constexpr std::uint32_t kCandidateDomain = 0x44494341u;
    RNG rng(restir_random_seed(seed, pixel, iteration, kCandidateDomain));
    for (std::uint32_t index = 0; index < candidate_count; ++index) {
        ++stats.attempted;
        RestirLightSample canonical;
        PackedLightSample evaluated;
        float selection_probability = 0.0f;
        const PackedLightStatus light_status =
            sample_restir_non_delta_light_core(
                scene, context.surface.position, rng, canonical, evaluated,
                selection_probability);
        if (light_status != PackedLightStatus::Success &&
            light_status != PackedLightStatus::NoSample) {
            ++stats.rejected;
            return RestirDIStatus::LightFailure;
        }

        Float3 integrand{};
        float target = 0.0f;
        RestirDIStatus evaluation_status = RestirDIStatus::Success;
        float proposal_pdf = selection_probability;
        if (light_status == PackedLightStatus::Success) {
            proposal_pdf *= evaluated.pdf;
            evaluation_status = evaluate_unshadowed_restir_direct(
                context, evaluated, integrand, target);
        }
        if (evaluation_status != RestirDIStatus::Success) {
            ++stats.rejected;
            return evaluation_status;
        }
        const bool nonzero =
            !packed_transport::math::black(integrand);
        const RestirDICandidate candidate = make_candidate(
            canonical, target, proposal_pdf, nonzero, canonical.valid());
        const ReservoirOperationResult update = stream_candidate(
            reservoir, candidate, static_cast<float>(rng.next()));
        stats.represented += update.represented_candidates;
        if (!update.accepted() &&
            update.rejection != ReservoirRejectReason::ZeroTarget) {
            ++stats.rejected;
            return RestirDIStatus::ReservoirFailure;
        }
    }
    const ReservoirOperationResult finalized = finalize_reservoir(reservoir);
    if (!finalized.accepted()) {
        return finalized.rejection == ReservoirRejectReason::EmptyReservoir ||
                       finalized.rejection ==
                           ReservoirRejectReason::NoSelectedSample
                   ? RestirDIStatus::ReservoirEmpty
                   : RestirDIStatus::ReservoirFailure;
    }
    return RestirDIStatus::Success;
}

RT_HOST_DEVICE RT_FORCE_INLINE RestirDIStatus shade_initial_di_reservoir(
    const CompiledSceneView &scene, const RestirSurface &surface,
    const RestirDIReservoir &reservoir, std::uint32_t width,
    std::uint32_t height, std::uint32_t pixel, std::uint32_t iteration,
    std::uint32_t seed, Float3 &radiance,
    std::uint32_t &visibility_rays) noexcept {
    radiance = {};
    visibility_rays = 0u;
    if (!surface.valid()) {
        RNG camera_rng(packed_transport::packed_camera_sample_seed(
            seed, pixel, iteration));
        const PackedRay camera_ray =
            packed_transport::generate_packed_camera_ray_core(
                scene.camera, pixel % width, pixel / width, width, height,
                camera_rng);
        radiance = scene.background;
        const std::uint32_t environment =
            packed_light::environment_light_id(scene);
        if (environment != kInvalidPackedIndex) {
            const PackedLightStatus environment_status =
                packed_light::environment_radiance_core(
                    scene, scene.lights[environment], camera_ray.direction,
                    radiance);
            if (environment_status != PackedLightStatus::Success &&
                environment_status != PackedLightStatus::NoSample) {
                return RestirDIStatus::LightFailure;
            }
        }
        return packed_transport::math::finite(radiance)
                   ? RestirDIStatus::NoSurface
                   : RestirDIStatus::NonFinite;
    }
    RestirDIPixelContext context;
    const RestirDIStatus context_status = reconstruct_restir_di_context(
        scene, surface, width, height, pixel, iteration, seed, context);
    if (context_status != RestirDIStatus::Success) {
        return context_status;
    }

    radiance = context.material.emission;
    constexpr std::uint32_t kVisibilityDomain = 0x44495649u;
    RNG visibility_rng(
        restir_random_seed(seed, pixel, iteration, kVisibilityDomain));
    for (std::uint32_t index = 0; index < scene.delta_light_indices.count;
         ++index) {
        const std::uint32_t light_id = scene.delta_light_indices[index];
        PackedLightSample light_sample{};
        const PackedLightStatus light_status =
            packed_light::sample_packed_light_core(
                scene, light_id, context.surface.position, {},
                light_sample);
        if (light_status == PackedLightStatus::NoSample) {
            continue;
        }
        if (light_status != PackedLightStatus::Success) {
            return RestirDIStatus::LightFailure;
        }
        Float3 contribution{};
        std::uint32_t shadow_rays = 0;
        const PackedTransportStatus transport_status =
            packed_transport::evaluate_direct_sample(
                scene, context.surface, context.material, context.wo,
                light_sample, 1.0f, false, surface.ray_time,
                visibility_rng, contribution, shadow_rays);
        visibility_rays += shadow_rays;
        if (transport_status != PackedTransportStatus::Success) {
            return transport_status ==
                           PackedTransportStatus::TraversalFailure
                       ? RestirDIStatus::TraversalFailure
                       : RestirDIStatus::LightFailure;
        }
        radiance = packed_transport::math::add(radiance, contribution);
    }

    if (!reservoir_is_usable(reservoir)) {
        return packed_transport::math::finite(radiance)
                   ? RestirDIStatus::ReservoirEmpty
                   : RestirDIStatus::NonFinite;
    }
    PackedLightSample selected;
    const PackedLightStatus selected_status =
        evaluate_restir_light_sample_core(
            scene, reservoir.sample, context.surface.position, selected);
    if (selected_status != PackedLightStatus::Success) {
        return selected_status == PackedLightStatus::NoSample
                   ? RestirDIStatus::ReservoirEmpty
                   : RestirDIStatus::LightFailure;
    }
    Float3 integrand{};
    float target = 0.0f;
    const RestirDIStatus target_status = evaluate_unshadowed_restir_direct(
        context, selected, integrand, target);
    if (target_status != RestirDIStatus::Success) {
        return target_status;
    }
    if (!(target > 0.0f) || packed_transport::math::black(integrand)) {
        return RestirDIStatus::ReservoirFailure;
    }
    ++visibility_rays;
    bool visible = false;
    const PackedTransportStatus visibility_status =
        packed_transport::visibility(
            scene, context.surface, selected, surface.ray_time,
            visibility_rng, visible);
    if (visibility_status != PackedTransportStatus::Success) {
        return RestirDIStatus::TraversalFailure;
    }
    if (visible) {
        radiance = packed_transport::math::add(
            radiance,
            packed_transport::math::multiply(
                integrand, reservoir.unbiased_contribution_weight));
    }
    return packed_transport::math::finite(radiance)
               ? RestirDIStatus::Success
               : RestirDIStatus::NonFinite;
}

} // namespace restir

#endif

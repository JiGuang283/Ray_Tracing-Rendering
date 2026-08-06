#ifndef RESTIR_GI_CORE_H
#define RESTIR_GI_CORE_H

#include "flat_intersector_core.h"
#include "packed_bsdf_core.h"
#include "packed_light_core.h"
#include "packed_material_core.h"
#include "packed_transport_core.h"
#include "restir_di_core.h"
#include "restir_gi_reservoir_core.h"
#include "surface_reconstruction_core.h"

#include <cfloat>
#include <cstdint>

namespace restir {

enum class RestirGIStatus : std::uint32_t {
    Success = 0,
    NoSurface = 1,
    UnsupportedPrimary = 2,
    PrimaryReconstructionFailure = 3,
    PrimaryMaterialFailure = 4,
    NoSecondary = 5,
    UnsupportedSecondary = 6,
    SecondaryReconstructionFailure = 7,
    SecondaryMaterialFailure = 8,
    BSDFFailure = 9,
    LightFailure = 10,
    TraversalFailure = 11,
    ReservoirEmpty = 12,
    ReservoirFailure = 13,
    VisibilityFailure = 14,
    NonFinite = 15,
};

enum class RestirGIShiftFailure : std::uint32_t {
    None = 0,
    InvalidSample = 1,
    UnsupportedPrimary = 2,
    UnsupportedSecondary = 3,
    BackfacingPrimary = 4,
    BackfacingSecondary = 5,
    DegenerateDistance = 6,
    MaterialMismatch = 7,
    Occluded = 8,
    NonFinite = 9,
    Count = 10,
};

struct RestirGICandidateStats {
    std::uint32_t attempted = 0u;
    std::uint32_t represented = 0u;
    std::uint32_t rejected = 0u;
    std::uint32_t suffix_shadow_rays = 0u;
    std::uint32_t suffix_traversal_steps = 0u;
    std::uint32_t fallback_paths = 0u;
};

struct RestirGIReconnectResult {
    Float3 integrand_area{};
    Float3 direction{};
    float distance = 0.0f;
    float target = 0.0f;
    RestirGIShiftFailure failure = RestirGIShiftFailure::None;
};

RT_HOST_DEVICE RT_FORCE_INLINE bool make_gi_visibility_ray(
    const PackedSurfaceInteraction &source, Float3 target_position,
    Float3 target_normal, Float3 direction, float distance, float time,
    PackedRay &ray) noexcept {
    if (!(distance > 0.0f) || !packed_transport::math::finite(distance) ||
        !packed_transport::math::finite(target_position) ||
        !packed_transport::math::finite(target_normal) ||
        !packed_transport::math::finite(direction)) {
        return false;
    }
    const float offset = distance * 0.25f < 1e-4f
                             ? distance * 0.25f
                             : 1e-4f;
    const float source_side =
        packed_transport::math::dot(direction,
                                    source.geometric_normal) >= 0.0f
            ? 1.0f
            : -1.0f;
    const float target_side =
        packed_transport::math::dot(
            packed_transport::math::multiply(direction, -1.0f),
            target_normal) >= 0.0f
            ? 1.0f
            : -1.0f;
    const Float3 origin = packed_transport::math::add(
        source.position,
        packed_transport::math::multiply(source.geometric_normal,
                                         source_side * offset));
    const Float3 target = packed_transport::math::add(
        target_position,
        packed_transport::math::multiply(target_normal,
                                         target_side * offset));
    const Float3 segment = packed_transport::math::add(
        target, packed_transport::math::multiply(origin, -1.0f));
    const float segment_length_squared =
        packed_transport::math::length_squared(segment);
    if (!(segment_length_squared > 0.0f) ||
        !packed_transport::math::finite(segment_length_squared)) {
        return false;
    }
    const float segment_length = ::sqrtf(segment_length_squared);
    ray = {};
    ray.origin = origin;
    ray.direction = packed_transport::math::multiply(
        segment, 1.0f / segment_length);
    ray.t_min = 0.0f;
    ray.t_max = segment_length * (1.0f - 1e-5f);
    ray.time = time;
    return ray.t_max > ray.t_min &&
           packed_transport::math::finite(ray.direction) &&
           packed_transport::math::finite(ray.t_max);
}

RT_HOST_DEVICE RT_FORCE_INLINE bool is_diffuse_secondary(
    const PackedMaterialOutput &material) noexcept {
    if (material.closure_count != 1u ||
        packed_transport::math::black(
            {material.closures[0].parameters.x,
             material.closures[0].parameters.y,
             material.closures[0].parameters.z})) {
        return false;
    }
    const PackedClosure &closure = material.closures[0];
    return closure.type == PackedClosureType::Lambertian &&
           closure.contribution_weight > 0.0f &&
           closure.sample_weight > 0.0f &&
           packed_transport::math::finite(closure.contribution_weight) &&
           packed_transport::math::finite(closure.sample_weight);
}

RT_HOST_DEVICE RT_FORCE_INLINE bool is_gi_primary_reconnectable(
    const PackedMaterialOutput &material) noexcept {
    bool has_continuous_surface = false;
    for (std::uint32_t index = 0u; index < material.closure_count; ++index) {
        const PackedClosure &closure = material.closures[index];
        if (!(closure.contribution_weight > 0.0f) ||
            !(closure.sample_weight > 0.0f) ||
            !packed_transport::math::finite(closure.contribution_weight) ||
            !packed_transport::math::finite(closure.sample_weight)) {
            continue;
        }
        switch (closure.type) {
        case PackedClosureType::Lambertian:
        case PackedClosureType::GGXReflection:
        case PackedClosureType::ClearcoatGGX:
            has_continuous_surface = true;
            break;
        case PackedClosureType::Mirror:
        case PackedClosureType::Dielectric:
        case PackedClosureType::IsotropicPhase:
            break;
        }
    }
    return has_continuous_surface;
}

RT_HOST_DEVICE RT_FORCE_INLINE bool solid_angle_pdf_to_area(
    float pdf_solid_angle, float squared_distance,
    float secondary_cosine, float &pdf_area) noexcept {
    pdf_area = 0.0f;
    if (!(pdf_solid_angle > 0.0f) || !(squared_distance > 0.0f) ||
        !(secondary_cosine > 0.0f) ||
        !packed_transport::math::finite(pdf_solid_angle) ||
        !packed_transport::math::finite(squared_distance) ||
        !packed_transport::math::finite(secondary_cosine)) {
        return false;
    }
    pdf_area = pdf_solid_angle * secondary_cosine / squared_distance;
    return pdf_area > 0.0f && packed_transport::math::finite(pdf_area);
}

RT_HOST_DEVICE RT_FORCE_INLINE RestirGIStatus reconstruct_restir_gi_context(
    const CompiledSceneView &scene, const RestirSurface &stored,
    const PackedCamera &camera, std::uint32_t width, std::uint32_t height,
    std::uint32_t pixel, std::uint32_t iteration, std::uint32_t seed,
    RestirDIPixelContext &context) noexcept {
    const RestirDIStatus status = reconstruct_restir_di_context(
        scene, stored, camera, width, height, pixel, iteration, seed,
        context);
    switch (status) {
    case RestirDIStatus::Success:
        return is_gi_primary_reconnectable(context.material)
                   ? RestirGIStatus::Success
                   : RestirGIStatus::UnsupportedPrimary;
    case RestirDIStatus::NoSurface:
        return RestirGIStatus::NoSurface;
    case RestirDIStatus::UnsupportedSurface:
        return RestirGIStatus::UnsupportedPrimary;
    case RestirDIStatus::ReconstructionFailure:
        return RestirGIStatus::PrimaryReconstructionFailure;
    case RestirDIStatus::MaterialFailure:
        return RestirGIStatus::PrimaryMaterialFailure;
    case RestirDIStatus::NonFinite:
        return RestirGIStatus::NonFinite;
    default:
        return RestirGIStatus::PrimaryReconstructionFailure;
    }
}

RT_HOST_DEVICE RT_FORCE_INLINE RestirGIStatus reconstruct_restir_gi_context(
    const CompiledSceneView &scene, const RestirSurface &stored,
    std::uint32_t width, std::uint32_t height, std::uint32_t pixel,
    std::uint32_t iteration, std::uint32_t seed,
    RestirDIPixelContext &context) noexcept {
    return reconstruct_restir_gi_context(
        scene, stored, scene.camera, width, height, pixel, iteration, seed,
        context);
}

RT_HOST_DEVICE RT_FORCE_INLINE RestirGIStatus evaluate_diffuse_suffix(
    const CompiledSceneView &scene,
    const PackedSurfaceInteraction &secondary,
    const PackedMaterialOutput &secondary_material, Float3 wo,
    float ray_time, const PackedTransportSettings &transport, RNG &rng,
    Float3 &radiance, std::uint32_t &shadow_rays,
    std::uint32_t &traversal_steps) noexcept {
    radiance = {};
    shadow_rays = 0u;
    traversal_steps = 0u;
    if (!is_diffuse_secondary(secondary_material)) {
        return RestirGIStatus::UnsupportedSecondary;
    }
    if (transport.max_depth < 2u ||
        !valid_integrator_policy(transport.policy)) {
        return RestirGIStatus::UnsupportedSecondary;
    }

    PackedTransportStatus direct_status =
        packed_transport::sample_direct_lighting(
            scene, secondary, secondary_material, wo,
            transport.max_depth > 2u, ray_time, rng, radiance,
            shadow_rays);
    if (direct_status != PackedTransportStatus::Success) {
        return direct_status == PackedTransportStatus::TraversalFailure
                   ? RestirGIStatus::TraversalFailure
                   : RestirGIStatus::LightFailure;
    }
    if (transport.max_depth <= 2u) {
        return packed_transport::math::finite(radiance)
                   ? RestirGIStatus::Success
                   : RestirGIStatus::NonFinite;
    }

    PackedBSDFSample sample{};
    const PackedBSDFStatus bsdf_status =
        packed_bsdf::sample_packed_bsdf_core(
            secondary_material, wo, rng, sample);
    if (bsdf_status == PackedBSDFStatus::Empty ||
        bsdf_status == PackedBSDFStatus::NoSample) {
        return RestirGIStatus::Success;
    }
    if (bsdf_status != PackedBSDFStatus::Success ||
        sample.is_delta() || !(sample.pdf >= 1e-8f)) {
        return RestirGIStatus::BSDFFailure;
    }
    const float cosine =
        packed_bsdf::abs_cos_theta(secondary_material, sample.wi);
    const Float3 throughput = packed_transport::math::multiply(
        sample.f, cosine / sample.pdf);
    if (!packed_transport::math::finite(throughput)) {
        return RestirGIStatus::NonFinite;
    }

    PackedPathState state{};
    const PackedRay ray = packed_transport::spawn_ray(
        secondary, sample.wi, ray_time);
    packed_transport::initialize_packed_path_state(
        scene, ray, transport, rng.state, 0u, 0u, state);
    if (!state.active()) {
        return RestirGIStatus::TraversalFailure;
    }
    state.depth = 2u;
    state.throughput = throughput;
    state.previous_bsdf_pdf = sample.pdf;
    state.flags &= ~PACKED_PATH_DELTA_BOUNCE;
    while (state.active()) {
        packed_transport::advance_packed_path_core(scene, transport, state);
    }
    rng.state = state.rng_state;
    shadow_rays += state.shadow_rays;
    traversal_steps += state.traversal_steps;
    if (state.status != PackedTransportStatus::Success) {
        return state.status == PackedTransportStatus::TraversalFailure
                   ? RestirGIStatus::TraversalFailure
                   : RestirGIStatus::NonFinite;
    }
    radiance = packed_transport::math::add(radiance, state.radiance);
    return packed_transport::math::finite(radiance)
               ? RestirGIStatus::Success
               : RestirGIStatus::NonFinite;
}

RT_HOST_DEVICE RT_FORCE_INLINE RestirGIStatus evaluate_gi_fallback_path(
    const CompiledSceneView &scene, const PackedRay &secondary_ray,
    const PackedMaterialOutput &primary_material,
    const PackedBSDFSample &primary_sample,
    const PackedTransportSettings &transport, RNG &rng, Float3 &radiance,
    std::uint32_t &shadow_rays,
    std::uint32_t &traversal_steps) noexcept {
    radiance = {};
    shadow_rays = 0u;
    traversal_steps = 0u;
    if (transport.max_depth < 2u ||
        !valid_integrator_policy(transport.policy) ||
        primary_sample.pdf <= 0.0f) {
        return RestirGIStatus::BSDFFailure;
    }
    const float cosine =
        packed_bsdf::abs_cos_theta(primary_material, primary_sample.wi);
    const Float3 throughput = packed_transport::math::multiply(
        primary_sample.f, cosine / primary_sample.pdf);
    if (!packed_transport::math::finite(throughput)) {
        return RestirGIStatus::NonFinite;
    }
    PackedPathState state{};
    packed_transport::initialize_packed_path_state(
        scene, secondary_ray, transport, rng.state, 0u, 0u, state);
    if (!state.active()) {
        return RestirGIStatus::TraversalFailure;
    }
    state.depth = 1u;
    state.throughput = throughput;
    state.previous_bsdf_pdf = primary_sample.pdf;
    if (primary_sample.is_delta()) {
        state.flags |= PACKED_PATH_DELTA_BOUNCE;
    } else {
        state.flags &= ~PACKED_PATH_DELTA_BOUNCE;
    }
    while (state.active()) {
        packed_transport::advance_packed_path_core(scene, transport, state);
    }
    rng.state = state.rng_state;
    shadow_rays = state.shadow_rays;
    traversal_steps = state.traversal_steps;
    if (state.status != PackedTransportStatus::Success) {
        return state.status == PackedTransportStatus::TraversalFailure
                   ? RestirGIStatus::TraversalFailure
                   : RestirGIStatus::NonFinite;
    }
    radiance = state.radiance;
    return packed_transport::math::finite(radiance)
               ? RestirGIStatus::Success
               : RestirGIStatus::NonFinite;
}

RT_HOST_DEVICE RT_FORCE_INLINE RestirGIStatus evaluate_diffuse_reconnection(
    const RestirDIPixelContext &destination,
    const RestirGISample &sample,
    RestirGIReconnectResult &result) noexcept {
    result = {};
    if (!sample.valid() ||
        !packed_transport::math::finite(sample.position) ||
        !packed_transport::math::finite(sample.suffix_radiance)) {
        result.failure = RestirGIShiftFailure::InvalidSample;
        return RestirGIStatus::ReservoirFailure;
    }
    if (!is_gi_primary_reconnectable(destination.material)) {
        result.failure = RestirGIShiftFailure::UnsupportedPrimary;
        return RestirGIStatus::UnsupportedPrimary;
    }
    const Float3 displacement = packed_transport::math::add(
        sample.position,
        packed_transport::math::multiply(destination.surface.position,
                                         -1.0f));
    const float distance_squared =
        packed_transport::math::length_squared(displacement);
    if (!(distance_squared > 1e-12f) ||
        !packed_transport::math::finite(distance_squared)) {
        result.failure = RestirGIShiftFailure::DegenerateDistance;
        return RestirGIStatus::ReservoirFailure;
    }
    result.distance = ::sqrtf(distance_squared);
    result.direction = packed_transport::math::multiply(
        displacement, 1.0f / result.distance);
    const Float3 secondary_normal =
        unpack_octahedral_normal(sample.geometric_normal);
    const float primary_cosine = packed_bsdf::abs_cos_theta(
        destination.material, result.direction);
    const float secondary_cosine = packed_transport::math::dot(
        secondary_normal,
        packed_transport::math::multiply(result.direction, -1.0f));
    if (!(primary_cosine > 0.0f)) {
        result.failure = RestirGIShiftFailure::BackfacingPrimary;
        return RestirGIStatus::Success;
    }
    if (!(secondary_cosine > 0.0f)) {
        result.failure = RestirGIShiftFailure::BackfacingSecondary;
        return RestirGIStatus::Success;
    }
    Float3 f{};
    const PackedBSDFStatus bsdf_status = packed_bsdf::eval_packed_bsdf_core(
        destination.material, destination.wo, result.direction, f);
    if (bsdf_status == PackedBSDFStatus::Empty) {
        result.failure = RestirGIShiftFailure::BackfacingPrimary;
        return RestirGIStatus::Success;
    }
    if (bsdf_status != PackedBSDFStatus::Success) {
        return RestirGIStatus::BSDFFailure;
    }
    const float geometry =
        primary_cosine * secondary_cosine / distance_squared;
    result.integrand_area = packed_transport::math::multiply(
        packed_transport::math::multiply(f, sample.suffix_radiance),
        geometry);
    result.target = packed_light::math::luminance(result.integrand_area);
    if (!packed_transport::math::finite(result.integrand_area) ||
        !packed_transport::math::finite(result.target)) {
        result = {};
        result.failure = RestirGIShiftFailure::NonFinite;
        return RestirGIStatus::NonFinite;
    }
    return RestirGIStatus::Success;
}

RT_HOST_DEVICE RT_FORCE_INLINE RestirGIStatus generate_initial_gi_reservoir(
    const CompiledSceneView &scene, const RestirSurface &stored,
    std::uint32_t width, std::uint32_t height, std::uint32_t pixel,
    std::uint32_t iteration, std::uint32_t seed,
    std::uint32_t candidate_count,
    const PackedTransportSettings &transport,
    RestirGIReservoir &reservoir,
    RestirGICandidateStats &stats, Float3 *fallback_radiance = nullptr) noexcept {
    reset_reservoir(reservoir);
    stats = {};
    if (fallback_radiance != nullptr) {
        *fallback_radiance = {};
    }
    if (candidate_count == 0u || transport.max_depth < 2u) {
        return RestirGIStatus::ReservoirEmpty;
    }
    RestirDIPixelContext primary;
    const RestirGIStatus context_status = reconstruct_restir_gi_context(
        scene, stored, width, height, pixel, iteration, seed, primary);
    if (context_status != RestirGIStatus::Success) {
        return context_status;
    }

    constexpr std::uint32_t kCandidateDomain = 0x47494341u;
    RNG rng(restir_random_seed(seed, pixel, iteration, kCandidateDomain));
    for (std::uint32_t index = 0u; index < candidate_count; ++index) {
        ++stats.attempted;
        PackedBSDFSample primary_sample{};
        const PackedBSDFStatus bsdf_status =
            packed_bsdf::sample_packed_bsdf_core(
                primary.material, primary.wo, rng, primary_sample);
        if (bsdf_status == PackedBSDFStatus::Empty ||
            bsdf_status == PackedBSDFStatus::NoSample) {
            const ReservoirOperationResult represented =
                represent_gi_candidates(reservoir, 1u, 1.0f);
            stats.represented += represented.represented_candidates;
            continue;
        }
        if (bsdf_status != PackedBSDFStatus::Success) {
            ++stats.rejected;
            return RestirGIStatus::BSDFFailure;
        }
        if (!(primary_sample.pdf >= 1e-8f)) {
            ++stats.rejected;
            return RestirGIStatus::BSDFFailure;
        }

        const PackedRay secondary_ray = packed_transport::spawn_ray(
            primary.surface, primary_sample.wi, stored.ray_time);
        if (primary_sample.is_delta()) {
            Float3 fallback{};
            std::uint32_t fallback_shadow_rays = 0u;
            std::uint32_t fallback_steps = 0u;
            const RestirGIStatus fallback_status = evaluate_gi_fallback_path(
                scene, secondary_ray, primary.material, primary_sample,
                transport, rng, fallback, fallback_shadow_rays,
                fallback_steps);
            stats.suffix_shadow_rays += fallback_shadow_rays;
            stats.suffix_traversal_steps += fallback_steps;
            if (fallback_status != RestirGIStatus::Success) {
                ++stats.rejected;
                return fallback_status;
            }
            if (fallback_radiance != nullptr) {
                *fallback_radiance = packed_transport::math::add(
                    *fallback_radiance,
                    packed_transport::math::multiply(
                        fallback, 1.0f / static_cast<float>(candidate_count)));
            }
            ++stats.fallback_paths;
            continue;
        }
        PackedHit secondary_hit{};
        const PackedTraversalStatus traversal =
            packed_intersector::intersect_compiled_scene_core(
                scene, secondary_ray, secondary_hit, &rng);
        ++stats.suffix_traversal_steps;
        if (traversal == PackedTraversalStatus::Miss) {
            const ReservoirOperationResult represented =
                represent_gi_candidates(reservoir, 1u, 1.0f);
            stats.represented += represented.represented_candidates;
            continue;
        }
        if (traversal != PackedTraversalStatus::Hit) {
            ++stats.rejected;
            return RestirGIStatus::TraversalFailure;
        }
        PackedSurfaceInteraction secondary{};
        const PackedShadingStatus reconstruction =
            packed_reconstruction::reconstruct_compiled_hit_core(
                scene, secondary_ray, secondary_hit, secondary);
        if (reconstruction != PackedShadingStatus::Success) {
            ++stats.rejected;
            return RestirGIStatus::SecondaryReconstructionFailure;
        }
        PackedMaterialOutput secondary_material{};
        const PackedShadingStatus material_status =
            packed_material::evaluate_packed_material_core(
                scene, secondary.material_id, secondary,
                secondary_material);
        if (material_status != PackedShadingStatus::Success) {
            ++stats.rejected;
            return RestirGIStatus::SecondaryMaterialFailure;
        }
        if (secondary_material.closure_count == 0u) {
            const ReservoirOperationResult represented =
                represent_gi_candidates(reservoir, 1u, 1.0f);
            stats.represented += represented.represented_candidates;
            continue;
        }
        if (!is_diffuse_secondary(secondary_material)) {
            Float3 fallback{};
            std::uint32_t fallback_shadow_rays = 0u;
            std::uint32_t fallback_steps = 0u;
            const RestirGIStatus fallback_status = evaluate_gi_fallback_path(
                scene, secondary_ray, primary.material, primary_sample,
                transport, rng, fallback, fallback_shadow_rays,
                fallback_steps);
            stats.suffix_shadow_rays += fallback_shadow_rays;
            stats.suffix_traversal_steps += fallback_steps;
            if (fallback_status != RestirGIStatus::Success) {
                ++stats.rejected;
                return fallback_status;
            }
            if (fallback_radiance != nullptr) {
                *fallback_radiance = packed_transport::math::add(
                    *fallback_radiance,
                    packed_transport::math::multiply(
                        fallback, 1.0f / static_cast<float>(candidate_count)));
            }
            ++stats.fallback_paths;
            continue;
        }

        const Float3 to_primary = packed_transport::math::multiply(
            primary_sample.wi, -1.0f);
        const float secondary_cosine = packed_transport::math::dot(
            secondary.geometric_normal, to_primary);
        const float distance_squared =
            packed_transport::math::length_squared(
                packed_transport::math::add(
                    secondary.position,
                    packed_transport::math::multiply(
                        primary.surface.position, -1.0f)));
        float proposal_pdf_area = 0.0f;
        if (!solid_angle_pdf_to_area(
                primary_sample.pdf, distance_squared, secondary_cosine,
                proposal_pdf_area)) {
            ++stats.rejected;
            return RestirGIStatus::NonFinite;
        }

        Float3 suffix{};
        std::uint32_t suffix_shadow_rays = 0u;
        std::uint32_t suffix_steps = 0u;
        const RestirGIStatus suffix_status = evaluate_diffuse_suffix(
            scene, secondary, secondary_material, to_primary,
            stored.ray_time, transport, rng, suffix,
            suffix_shadow_rays, suffix_steps);
        stats.suffix_shadow_rays += suffix_shadow_rays;
        stats.suffix_traversal_steps += suffix_steps;
        if (suffix_status != RestirGIStatus::Success) {
            ++stats.rejected;
            return suffix_status;
        }

        RestirGISample canonical;
        canonical.position = secondary.position;
        canonical.source_pdf_area = proposal_pdf_area;
        canonical.suffix_radiance = suffix;
        canonical.geometric_normal =
            pack_octahedral_normal(secondary.geometric_normal);
        canonical.material_id = secondary.material_id;
        canonical.instance_id = secondary_hit.instance_id;
        canonical.primitive_id = secondary_hit.primitive_id;
        canonical.source_pixel = pixel;
        RestirGIReconnectResult evaluated;
        const RestirGIStatus reconnect_status =
            evaluate_diffuse_reconnection(primary, canonical, evaluated);
        if (reconnect_status != RestirGIStatus::Success) {
            ++stats.rejected;
            return reconnect_status;
        }
        const bool nonzero = !packed_transport::math::black(
            evaluated.integrand_area);
        const RestirGICandidate candidate = make_candidate(
            canonical, evaluated.target, proposal_pdf_area, nonzero,
            canonical.valid());
        const ReservoirOperationResult update = stream_candidate(
            reservoir, candidate, static_cast<float>(rng.next()));
        stats.represented += update.represented_candidates;
        if (!update.accepted() &&
            update.rejection != ReservoirRejectReason::ZeroTarget) {
            ++stats.rejected;
            return RestirGIStatus::ReservoirFailure;
        }
    }
    const ReservoirOperationResult finalized = finalize_reservoir(reservoir);
    if (!finalized.accepted()) {
        return finalized.rejection == ReservoirRejectReason::EmptyReservoir ||
                       finalized.rejection ==
                           ReservoirRejectReason::NoSelectedSample
                   ? RestirGIStatus::ReservoirEmpty
                   : RestirGIStatus::ReservoirFailure;
    }
    return RestirGIStatus::Success;
}

RT_HOST_DEVICE RT_FORCE_INLINE RestirGIStatus shade_gi_reservoir(
    const CompiledSceneView &scene, const RestirSurface &stored,
    const RestirGIReservoir &reservoir, std::uint32_t width,
    std::uint32_t height, std::uint32_t pixel,
    std::uint32_t iteration, std::uint32_t seed, Float3 &radiance,
    std::uint32_t &visibility_rays,
    RestirGIShiftFailure &failure) noexcept {
    radiance = {};
    visibility_rays = 0u;
    failure = RestirGIShiftFailure::None;
    if (!reservoir_is_usable(reservoir)) {
        return RestirGIStatus::ReservoirEmpty;
    }
    RestirDIPixelContext primary;
    const RestirGIStatus context_status = reconstruct_restir_gi_context(
        scene, stored, width, height, pixel, iteration, seed, primary);
    if (context_status != RestirGIStatus::Success) {
        return context_status;
    }
    RestirGIReconnectResult reconnect;
    const RestirGIStatus reconnect_status =
        evaluate_diffuse_reconnection(primary, reservoir.sample, reconnect);
    failure = reconnect.failure;
    if (reconnect_status != RestirGIStatus::Success ||
        !(reconnect.target > 0.0f)) {
        return reconnect_status;
    }

    ++visibility_rays;
    constexpr std::uint32_t kVisibilityDomain = 0x47495649u;
    RNG visibility_rng(
        restir_random_seed(seed, pixel, iteration, kVisibilityDomain));
    const Float3 secondary_normal =
        unpack_octahedral_normal(reservoir.sample.geometric_normal);
    PackedRay shadow;
    if (!make_gi_visibility_ray(
            primary.surface, reservoir.sample.position, secondary_normal,
            reconnect.direction, reconnect.distance, stored.ray_time,
            shadow)) {
        failure = RestirGIShiftFailure::DegenerateDistance;
        return RestirGIStatus::VisibilityFailure;
    }
    PackedHit blocker{};
    const PackedTraversalStatus visibility =
        packed_intersector::intersect_compiled_scene_core(
            scene, shadow, blocker, &visibility_rng);
    if (visibility == PackedTraversalStatus::Hit) {
        failure = RestirGIShiftFailure::Occluded;
        return RestirGIStatus::Success;
    }
    if (visibility != PackedTraversalStatus::Miss) {
        return RestirGIStatus::TraversalFailure;
    }
    radiance = packed_transport::math::multiply(
        reconnect.integrand_area,
        reservoir.unbiased_contribution_weight);
    return packed_transport::math::finite(radiance)
               ? RestirGIStatus::Success
               : RestirGIStatus::NonFinite;
}

static_assert(static_cast<std::uint32_t>(RestirGIShiftFailure::Count) <=
              kRestirShiftFailureBuckets);

} // namespace restir

#endif

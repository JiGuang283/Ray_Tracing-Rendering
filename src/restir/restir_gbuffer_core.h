#ifndef RESTIR_GBUFFER_CORE_H
#define RESTIR_GBUFFER_CORE_H

#include "flat_intersector_core.h"
#include "packed_material_core.h"
#include "packed_transport_core.h"
#include "restir_surface.h"
#include "surface_reconstruction_core.h"

namespace restir {
namespace gbuffer_detail {

RT_HOST_DEVICE RT_FORCE_INLINE bool closure_is_delta(
    PackedClosureType type) noexcept {
    return type == PackedClosureType::Mirror ||
           type == PackedClosureType::Dielectric;
}

RT_HOST_DEVICE RT_FORCE_INLINE float primary_view_depth(
    const PackedCamera &camera, Float3 position) noexcept {
    using namespace packed_transport::math;
    const Float3 image_center =
        add(camera.lower_left_corner,
            add(multiply(camera.horizontal, 0.5f),
                multiply(camera.vertical, 0.5f)));
    const Float3 forward =
        normalize(add(image_center, multiply(camera.origin, -1.0f)));
    return dot(add(position, multiply(camera.origin, -1.0f)), forward);
}

} // namespace gbuffer_detail

RT_HOST_DEVICE RT_FORCE_INLINE RestirGBufferStatus build_primary_surface_core(
    const CompiledSceneView &scene, std::uint32_t width,
    std::uint32_t height, std::uint32_t pixel_index,
    std::uint32_t iteration, std::uint32_t seed,
    RestirSurface &output) noexcept {
    output = {};
    output.motion = kInvalidPackedIndex;
    const std::uint64_t pixel_count =
        static_cast<std::uint64_t>(width) * height;
    if (width < 2u || height < 2u || pixel_index >= pixel_count) {
        return RestirGBufferStatus::InvalidInput;
    }

    RNG rng(packed_transport::packed_camera_sample_seed(seed, pixel_index,
                                                        iteration));
    const PackedRay ray =
        packed_transport::generate_packed_camera_ray_core(
            scene.camera, pixel_index % width, pixel_index / width, width,
            height, rng);
    PackedHit hit{};
    const PackedTraversalStatus traversal =
        packed_intersector::intersect_compiled_scene_core(scene, ray, hit,
                                                          &rng);
    if (traversal == PackedTraversalStatus::Miss) {
        output.ray_time = ray.time;
        return RestirGBufferStatus::Miss;
    }
    if (traversal != PackedTraversalStatus::Hit) {
        return traversal == PackedTraversalStatus::InvalidInput
                   ? RestirGBufferStatus::InvalidInput
                   : RestirGBufferStatus::TraversalFailure;
    }

    PackedSurfaceInteraction surface{};
    const PackedShadingStatus reconstruction =
        packed_reconstruction::reconstruct_compiled_hit_core(
            scene, ray, hit, surface);
    if (reconstruction != PackedShadingStatus::Success) {
        return reconstruction == PackedShadingStatus::NonFinite
                   ? RestirGBufferStatus::NonFinite
                   : RestirGBufferStatus::ReconstructionFailure;
    }
    PackedMaterialOutput material{};
    const PackedShadingStatus material_status =
        packed_material::evaluate_packed_material_core(
            scene, surface.material_id, surface, material);
    if (material_status != PackedShadingStatus::Success) {
        return material_status == PackedShadingStatus::NonFinite
                   ? RestirGBufferStatus::NonFinite
                   : RestirGBufferStatus::MaterialFailure;
    }

    output.position = surface.position;
    output.view_depth =
        gbuffer_detail::primary_view_depth(scene.camera, surface.position);
    output.hit_t = hit.t;
    output.barycentric_u = hit.barycentric_u;
    output.barycentric_v = hit.barycentric_v;
    output.ray_time = ray.time;
    output.instance_id = hit.instance_id;
    output.primitive_id = hit.primitive_id;
    output.material_id = surface.material_id;
    output.emitter_id = surface.emitter_id;
    output.geometric_normal =
        pack_octahedral_normal(surface.geometric_normal);
    output.shading_normal =
        pack_octahedral_normal(material.frame.normal);
    output.flags = (hit.flags & RESTIR_SURFACE_HIT_FLAGS_MASK) |
                   RESTIR_SURFACE_VALID;
    if ((hit.flags & PACKED_HIT_MEDIUM) != 0u) {
        output.flags |= RESTIR_SURFACE_UNSUPPORTED_DOMAIN;
    }
    if (material.closure_count == 0u) {
        output.flags |= RESTIR_SURFACE_NO_SCATTER;
    } else {
        bool delta_only = true;
        for (std::uint32_t index = 0; index < material.closure_count;
             ++index) {
            delta_only = delta_only &&
                         gbuffer_detail::closure_is_delta(
                             material.closures[index].type);
        }
        if (delta_only) {
            output.flags |= RESTIR_SURFACE_DELTA_ONLY;
        }
    }
    if (!surface_detail::finite(output.position) ||
        !surface_detail::finite(output.view_depth) ||
        !surface_detail::finite(output.hit_t) ||
        !surface_detail::finite(output.ray_time)) {
        output = {};
        return RestirGBufferStatus::NonFinite;
    }
    return RestirGBufferStatus::Success;
}

} // namespace restir

#endif

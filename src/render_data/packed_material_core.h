#ifndef PACKED_MATERIAL_CORE_H
#define PACKED_MATERIAL_CORE_H

#include "flat_intersector_core.h"
#include "packed_material.h"
#include "packed_texture_core.h"

#include <cmath>
#include <cstdint>

namespace packed_material {

namespace math {

RT_HOST_DEVICE RT_FORCE_INLINE float minimum(float a, float b) {
    return a < b ? a : b;
}

RT_HOST_DEVICE RT_FORCE_INLINE float maximum(float a, float b) {
    return a > b ? a : b;
}

RT_HOST_DEVICE RT_FORCE_INLINE float clamp(float value, float lower,
                                           float upper) {
    return maximum(lower, minimum(value, upper));
}

RT_HOST_DEVICE RT_FORCE_INLINE float absolute(float value) {
    return value < 0.0f ? -value : value;
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 add(Float3 a, Float3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 subtract(Float3 a, Float3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 multiply(Float3 value, float scalar) {
    return {value.x * scalar, value.y * scalar, value.z * scalar};
}

RT_HOST_DEVICE RT_FORCE_INLINE float dot(Float3 a, Float3 b) {
    return ::fmaf(a.x, b.x, ::fmaf(a.y, b.y, a.z * b.z));
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 cross(Float3 a, Float3 b) {
    return {::fmaf(a.y, b.z, -a.z * b.y),
            ::fmaf(a.z, b.x, -a.x * b.z),
            ::fmaf(a.x, b.y, -a.y * b.x)};
}

RT_HOST_DEVICE RT_FORCE_INLINE float length(Float3 value) {
    return ::sqrtf(dot(value, value));
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 normalize(Float3 value) {
    const float magnitude = length(value);
    if (!(magnitude > 0.0f) || !packed_intersector::finite(magnitude)) {
        return {};
    }
    return multiply(value, 1.0f / magnitude);
}

RT_HOST_DEVICE RT_FORCE_INLINE bool near_zero(Float3 value) {
    return absolute(value.x) < 1e-8f && absolute(value.y) < 1e-8f &&
           absolute(value.z) < 1e-8f;
}

RT_HOST_DEVICE RT_FORCE_INLINE bool finite(Float3 value) {
    return packed_intersector::finite(value.x) &&
           packed_intersector::finite(value.y) &&
           packed_intersector::finite(value.z);
}

RT_HOST_DEVICE RT_FORCE_INLINE bool finite(Float4 value) {
    return finite(Float3{value.x, value.y, value.z}) &&
           packed_intersector::finite(value.w);
}

} // namespace math

RT_HOST_DEVICE RT_FORCE_INLINE Float3 frame_bitangent(
    const PackedShadingFrame &frame) {
    return math::multiply(math::normalize(math::cross(frame.normal,
                                                       frame.tangent)),
                          frame.handedness);
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 frame_to_world(
    const PackedShadingFrame &frame, Float3 local) {
    const Float3 bitangent = frame_bitangent(frame);
    return math::add(math::multiply(frame.tangent, local.x),
                     math::add(math::multiply(bitangent, local.y),
                               math::multiply(frame.normal, local.z)));
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedShadingFrame frame_from_normal(
    Float3 normal) {
    PackedShadingFrame frame;
    frame.normal = math::normalize(normal);
    const Float3 helper = math::absolute(frame.normal.x) > 0.9f
                              ? Float3{0.0f, 1.0f, 0.0f}
                              : Float3{1.0f, 0.0f, 0.0f};
    const Float3 bitangent =
        math::normalize(math::cross(frame.normal, helper));
    frame.tangent = math::cross(bitangent, frame.normal);
    frame.handedness = 1.0f;
    return frame;
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedShadingFrame frame_from_tangent_space(
    Float3 normal, Float3 dpdu, Float3 dpdv) {
    PackedShadingFrame frame;
    frame.normal = math::normalize(normal);
    frame.tangent = math::subtract(
        dpdu, math::multiply(frame.normal, math::dot(dpdu, frame.normal)));
    if (math::near_zero(frame.tangent)) {
        frame.tangent = math::cross(dpdv, frame.normal);
    }
    if (math::near_zero(frame.tangent)) {
        return frame_from_normal(frame.normal);
    }
    frame.tangent = math::normalize(frame.tangent);
    const Float3 canonical_bitangent =
        math::normalize(math::cross(frame.normal, frame.tangent));
    frame.handedness =
        math::dot(canonical_bitangent, dpdv) < 0.0f ? -1.0f : 1.0f;
    return frame;
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedTextureEvalContext texture_context(
    const PackedSurfaceInteraction &surface) {
    PackedTextureEvalContext context;
    context.position = surface.position;
    context.uv0 = surface.uv;
    context.vertex_color = surface.vertex_color;
    return context;
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedShadingStatus evaluate_texture(
    const CompiledSceneView &scene, std::uint32_t texture_id,
    const PackedTextureEvalContext &context, Float4 &sample,
    std::uint32_t &max_texture_stack) {
    std::uint32_t used = 0;
    const PackedShadingStatus status =
        packed_texture::evaluate_packed_texture_core(
            scene, texture_id, context, sample, &used);
    if (used > max_texture_stack) {
        max_texture_stack = used;
    }
    return status;
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedShadingStatus apply_normal_map(
    const CompiledSceneView &scene, const PackedMaterial &material,
    const PackedSurfaceInteraction &surface,
    const PackedTextureEvalContext &context, PackedShadingFrame &frame,
    std::uint32_t &max_texture_stack) {
    frame = frame_from_tangent_space(surface.shading_normal, surface.dpdu,
                                     surface.dpdv);
    const std::uint32_t texture_id = material.texture_ids[3];
    if (texture_id == kInvalidPackedIndex) {
        return PackedShadingStatus::Success;
    }
    Float4 encoded{};
    const PackedShadingStatus status = evaluate_texture(
        scene, texture_id, context, encoded, max_texture_stack);
    if (status != PackedShadingStatus::Success) {
        return status;
    }
    const float strength = math::maximum(0.0f, material.parameters[0].y);
    Float3 local{(2.0f * encoded.x - 1.0f) * strength,
                 (2.0f * encoded.y - 1.0f) * strength,
                 2.0f * encoded.z - 1.0f};
    if ((material.flags & PACKED_MATERIAL_NORMAL_DIRECTX) != 0) {
        local.y = -local.y;
    }
    if (math::near_zero(local)) {
        return PackedShadingStatus::Success;
    }
    Float3 mapped = math::normalize(
        frame_to_world(frame, math::normalize(local)));
    if (math::dot(mapped, surface.geometric_normal) < 0.0f) {
        mapped = math::multiply(mapped, -1.0f);
    }
    frame = frame_from_tangent_space(mapped, frame.tangent,
                                     frame_bitangent(frame));
    return math::finite(frame.normal) && math::finite(frame.tangent)
               ? PackedShadingStatus::Success
               : PackedShadingStatus::NonFinite;
}

RT_HOST_DEVICE RT_FORCE_INLINE bool black(Float3 value) {
    return value.x <= 0.0f && value.y <= 0.0f && value.z <= 0.0f;
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedShadingStatus add_closure(
    PackedMaterialOutput &output, PackedClosureType type, Float4 parameters,
    float contribution_weight, float sample_weight,
    std::uint32_t flags = PACKED_CLOSURE_NONE) {
    if (sample_weight <= 0.0f || contribution_weight <= 0.0f) {
        return PackedShadingStatus::Success;
    }
    if (output.closure_count >= PackedMaterialOutput::kMaxClosures) {
        return PackedShadingStatus::ClosureOverflow;
    }
    PackedClosure &closure = output.closures[output.closure_count++];
    closure.parameters = parameters;
    closure.type = type;
    closure.flags = flags;
    closure.contribution_weight = contribution_weight;
    closure.sample_weight = sample_weight;
    return PackedShadingStatus::Success;
}

RT_HOST_DEVICE RT_FORCE_INLINE bool finite(
    const PackedMaterialOutput &output) {
    if (!math::finite(output.frame.normal) ||
        !math::finite(output.frame.tangent) ||
        !math::finite(output.geometry_normal) ||
        !math::finite(output.emission) ||
        !packed_intersector::finite(output.opacity) ||
        !packed_intersector::finite(output.frame.handedness)) {
        return false;
    }
    for (std::uint32_t index = 0; index < output.closure_count; ++index) {
        const PackedClosure &closure = output.closures[index];
        if (!math::finite(closure.parameters) ||
            !packed_intersector::finite(closure.contribution_weight) ||
            !packed_intersector::finite(closure.sample_weight)) {
            return false;
        }
    }
    return true;
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedShadingStatus
evaluate_packed_material_core(const CompiledSceneView &scene,
                              std::uint32_t material_id,
                              const PackedSurfaceInteraction &surface,
                              PackedMaterialOutput &output,
                              std::uint32_t *reported_texture_stack =
                                  nullptr) {
    if (material_id >= scene.materials.count) {
        return PackedShadingStatus::InvalidMaterial;
    }
    output = {};
    output.geometry_normal = surface.geometric_normal;
    output.frame = frame_from_tangent_space(
        surface.shading_normal, surface.dpdu, surface.dpdv);
    const PackedMaterial &material = scene.materials[material_id];
    const PackedTextureEvalContext context = texture_context(surface);
    const bool front_face =
        (surface.flags & PACKED_HIT_FRONT_FACE) != 0;
    std::uint32_t max_texture_stack = 0;
    PackedShadingStatus status = PackedShadingStatus::Success;
    Float4 texture{};

    switch (material.type) {
    case PackedMaterialType::Lambertian:
        status = evaluate_texture(scene, material.texture_ids[0], context,
                                  texture, max_texture_stack);
        if (status == PackedShadingStatus::Success &&
            !black({texture.x, texture.y, texture.z})) {
            status = add_closure(output, PackedClosureType::Lambertian,
                                 texture, 1.0f, 1.0f);
        }
        break;
    case PackedMaterialType::Metal: {
        const Float4 parameters = material.parameters[0];
        const float roughness = math::clamp(parameters.w, 0.0f, 1.0f);
        if (!black({parameters.x, parameters.y, parameters.z})) {
            status = add_closure(
                output,
                roughness <= 0.001f ? PackedClosureType::Mirror
                                     : PackedClosureType::GGXReflection,
                {parameters.x, parameters.y, parameters.z, roughness},
                1.0f, 1.0f);
        }
        break;
    }
    case PackedMaterialType::Dielectric:
        status = add_closure(
            output, PackedClosureType::Dielectric,
            {math::maximum(material.parameters[0].x, 1.0001f), 0.0f,
             0.0f, 0.0f},
            1.0f, 1.0f,
            front_face ? PACKED_CLOSURE_FRONT_FACE : PACKED_CLOSURE_NONE);
        break;
    case PackedMaterialType::DiffuseLight:
        if (front_face) {
            status = evaluate_texture(scene, material.texture_ids[0],
                                      context, texture,
                                      max_texture_stack);
            if (status == PackedShadingStatus::Success) {
                output.emission = {texture.x, texture.y, texture.z};
            }
        }
        break;
    case PackedMaterialType::Principled: {
        status = apply_normal_map(scene, material, surface, context,
                                  output.frame, max_texture_stack);
        if (status != PackedShadingStatus::Success) {
            break;
        }
        Float4 base{};
        Float4 roughness_sample{};
        Float4 metallic_sample{};
        status = evaluate_texture(scene, material.texture_ids[0], context,
                                  base, max_texture_stack);
        if (status != PackedShadingStatus::Success) {
            break;
        }
        status = evaluate_texture(scene, material.texture_ids[1], context,
                                  roughness_sample, max_texture_stack);
        if (status != PackedShadingStatus::Success) {
            break;
        }
        status = evaluate_texture(scene, material.texture_ids[2], context,
                                  metallic_sample, max_texture_stack);
        if (status != PackedShadingStatus::Success) {
            break;
        }
        const float roughness =
            math::clamp(roughness_sample.x, 0.01f, 1.0f);
        const float metallic =
            math::clamp(metallic_sample.x, 0.0f, 1.0f);
        const float diffuse_weight = 1.0f - metallic;
        const Float3 base_color{base.x, base.y, base.z};
        const Float3 diffuse = math::multiply(base_color, diffuse_weight);
        if (diffuse_weight > 0.0f && !black(diffuse)) {
            status = add_closure(
                output, PackedClosureType::Lambertian,
                {diffuse.x, diffuse.y, diffuse.z, 0.0f}, 1.0f,
                diffuse_weight);
        }
        if (status != PackedShadingStatus::Success) {
            break;
        }
        const Float3 dielectric_f0{0.04f, 0.04f, 0.04f};
        const Float3 f0 = math::add(
            math::multiply(dielectric_f0, 1.0f - metallic),
            math::multiply(base_color, metallic));
        status = add_closure(
            output, PackedClosureType::GGXReflection,
            {f0.x, f0.y, f0.z, roughness}, 1.0f,
            0.5f + 0.5f * metallic);
        if (status != PackedShadingStatus::Success) {
            break;
        }
        if (material.texture_ids[5] != kInvalidPackedIndex) {
            Float4 clearcoat{};
            status = evaluate_texture(
                scene, material.texture_ids[5], context, clearcoat,
                max_texture_stack);
            if (status != PackedShadingStatus::Success) {
                break;
            }
            const float strength = math::clamp(clearcoat.x, 0.0f, 1.0f);
            if (strength > 0.0f) {
                float coat_roughness = 0.1f;
                if (material.texture_ids[6] != kInvalidPackedIndex) {
                    Float4 coat_sample{};
                    status = evaluate_texture(
                        scene, material.texture_ids[6], context,
                        coat_sample, max_texture_stack);
                    if (status != PackedShadingStatus::Success) {
                        break;
                    }
                    coat_roughness =
                        math::clamp(coat_sample.x, 0.01f, 1.0f);
                }
                status = add_closure(
                    output, PackedClosureType::ClearcoatGGX,
                    {0.04f, 0.04f, 0.04f, coat_roughness}, strength,
                    strength);
                if (status != PackedShadingStatus::Success) {
                    break;
                }
            }
        }
        if (material.texture_ids[4] != kInvalidPackedIndex &&
            (front_face ||
             (material.flags & PACKED_MATERIAL_DOUBLE_SIDED) != 0)) {
            Float4 emission{};
            status = evaluate_texture(
                scene, material.texture_ids[4], context, emission,
                max_texture_stack);
            if (status == PackedShadingStatus::Success) {
                const float strength = material.parameters[0].x;
                output.emission = {emission.x * strength,
                                   emission.y * strength,
                                   emission.z * strength};
            }
        }
        break;
    }
    case PackedMaterialType::Isotropic:
        status = evaluate_texture(scene, material.texture_ids[0], context,
                                  texture, max_texture_stack);
        if (status == PackedShadingStatus::Success &&
            !black({texture.x, texture.y, texture.z})) {
            status = add_closure(output, PackedClosureType::IsotropicPhase,
                                 texture, 1.0f, 1.0f);
        }
        break;
    }

    if (reported_texture_stack != nullptr) {
        *reported_texture_stack = max_texture_stack;
    }
    if (status != PackedShadingStatus::Success) {
        return status;
    }
    return finite(output) ? PackedShadingStatus::Success
                          : PackedShadingStatus::NonFinite;
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedShadingStatus
evaluate_packed_material_emission_core(
    const CompiledSceneView &scene, std::uint32_t material_id,
    const PackedTextureEvalContext &context, Float3 &emission) {
    emission = {};
    if (material_id >= scene.materials.count) {
        return PackedShadingStatus::InvalidMaterial;
    }
    const PackedMaterial &material = scene.materials[material_id];
    if ((material.flags & PACKED_MATERIAL_EMISSIVE) == 0) {
        return PackedShadingStatus::Success;
    }

    std::uint32_t texture_id = kInvalidPackedIndex;
    float strength = 1.0f;
    if (material.type == PackedMaterialType::DiffuseLight) {
        texture_id = material.texture_ids[0];
    } else if (material.type == PackedMaterialType::Principled) {
        texture_id = material.texture_ids[4];
        strength = material.parameters[0].x;
    } else {
        emission = {material.emission_estimate.x,
                    material.emission_estimate.y,
                    material.emission_estimate.z};
        return math::finite(emission) ? PackedShadingStatus::Success
                                      : PackedShadingStatus::NonFinite;
    }

    Float4 texture{};
    const PackedShadingStatus status =
        packed_texture::evaluate_packed_texture_core(
            scene, texture_id, context, texture);
    if (status != PackedShadingStatus::Success) {
        return status;
    }
    emission = {texture.x * strength, texture.y * strength,
                texture.z * strength};
    return math::finite(emission) ? PackedShadingStatus::Success
                                  : PackedShadingStatus::NonFinite;
}

} // namespace packed_material

#endif

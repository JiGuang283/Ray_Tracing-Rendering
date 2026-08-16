#ifndef PACKED_BSDF_CORE_H
#define PACKED_BSDF_CORE_H

#include "packed_types.h"
#include "rng.h"

#include <cfloat>
#include <cmath>
#include <cstdint>

namespace packed_bsdf {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kInvPi = 1.0f / kPi;
constexpr float kInvFourPi = 1.0f / (4.0f * kPi);

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

RT_HOST_DEVICE RT_FORCE_INLINE bool finite(float value) {
    return value == value && value <= FLT_MAX && value >= -FLT_MAX;
}

RT_HOST_DEVICE RT_FORCE_INLINE bool finite(Float3 value) {
    return finite(value.x) && finite(value.y) && finite(value.z);
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

RT_HOST_DEVICE RT_FORCE_INLINE Float3 multiply(Float3 a, Float3 b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

RT_HOST_DEVICE RT_FORCE_INLINE float dot(Float3 a, Float3 b) {
    return ::fmaf(a.x, b.x, ::fmaf(a.y, b.y, a.z * b.z));
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 cross(Float3 a, Float3 b) {
    return {::fmaf(a.y, b.z, -a.z * b.y),
            ::fmaf(a.z, b.x, -a.x * b.z),
            ::fmaf(a.x, b.y, -a.y * b.x)};
}

RT_HOST_DEVICE RT_FORCE_INLINE float length_squared(Float3 value) {
    return dot(value, value);
}

RT_HOST_DEVICE RT_FORCE_INLINE float length(Float3 value) {
    return ::sqrtf(length_squared(value));
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 normalize(Float3 value) {
    const float magnitude = length(value);
    if (!(magnitude > 0.0f) || !finite(magnitude)) {
        return {};
    }
    return multiply(value, 1.0f / magnitude);
}

RT_HOST_DEVICE RT_FORCE_INLINE bool near_zero(Float3 value) {
    return absolute(value.x) < 1e-8f && absolute(value.y) < 1e-8f &&
           absolute(value.z) < 1e-8f;
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 reflect(Float3 value,
                                              Float3 normal) {
    return subtract(value, multiply(normal, 2.0f * dot(value, normal)));
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 refract(Float3 incident,
                                              Float3 normal,
                                              float eta_ratio) {
    const Float3 negative_incident = multiply(incident, -1.0f);
    const float cosine = minimum(dot(negative_incident, normal), 1.0f);
    const Float3 perpendicular =
        multiply(add(incident, multiply(normal, cosine)), eta_ratio);
    const float parallel_length =
        ::sqrtf(absolute(1.0f - length_squared(perpendicular)));
    return add(perpendicular, multiply(normal, -parallel_length));
}

} // namespace math

RT_HOST_DEVICE RT_FORCE_INLINE Float3 frame_bitangent(
    const PackedShadingFrame &frame) {
    return math::multiply(
        math::normalize(math::cross(frame.normal, frame.tangent)),
        frame.handedness);
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 frame_to_world(
    const PackedShadingFrame &frame, Float3 local) {
    const Float3 bitangent = frame_bitangent(frame);
    return math::add(math::multiply(frame.tangent, local.x),
                     math::add(math::multiply(bitangent, local.y),
                               math::multiply(frame.normal, local.z)));
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 frame_to_local(
    const PackedShadingFrame &frame, Float3 world) {
    const Float3 bitangent = frame_bitangent(frame);
    return {math::dot(world, frame.tangent), math::dot(world, bitangent),
            math::dot(world, frame.normal)};
}

RT_HOST_DEVICE RT_FORCE_INLINE bool closure_is_delta(
    PackedClosureType type) {
    return type == PackedClosureType::Mirror ||
           type == PackedClosureType::Dielectric;
}

RT_HOST_DEVICE RT_FORCE_INLINE bool closure_is_phase(
    PackedClosureType type) {
    return type == PackedClosureType::IsotropicPhase;
}

RT_HOST_DEVICE RT_FORCE_INLINE bool output_is_phase(
    const PackedMaterialOutput &output) {
    return output.closure_count == 1 &&
           closure_is_phase(output.closures[0].type);
}

RT_HOST_DEVICE RT_FORCE_INLINE float total_sample_weight(
    const PackedMaterialOutput &output) {
    float total = 0.0f;
    const std::uint32_t count =
        output.closure_count < PackedMaterialOutput::kMaxClosures
            ? output.closure_count
            : PackedMaterialOutput::kMaxClosures;
    for (std::uint32_t index = 0; index < count; ++index) {
        total += math::maximum(output.closures[index].sample_weight, 0.0f);
    }
    return total;
}

RT_HOST_DEVICE RT_FORCE_INLINE bool valid_event(
    const PackedMaterialOutput &output, Float3 wo, Float3 wi,
    std::uint32_t flags) {
    if ((flags & PACKED_BSDF_PHASE) != 0) {
        return true;
    }
    const float outgoing_side = math::dot(wo, output.geometry_normal);
    const float incoming_side = math::dot(wi, output.geometry_normal);
    if (math::absolute(outgoing_side) <= 1e-8f ||
        math::absolute(incoming_side) <= 1e-8f) {
        return false;
    }
    if ((flags & PACKED_BSDF_TRANSMISSION) != 0) {
        return outgoing_side * incoming_side < 0.0f;
    }
    return outgoing_side * incoming_side > 0.0f;
}

RT_HOST_DEVICE RT_FORCE_INLINE float saturate(float value) {
    return math::clamp(value, 0.0f, 1.0f);
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 fresnel_schlick(float cosine,
                                                      Float3 f0) {
    const float one_minus = 1.0f - saturate(cosine);
    const float square = one_minus * one_minus;
    const float factor = square * square * one_minus;
    return {f0.x + (1.0f - f0.x) * factor,
            f0.y + (1.0f - f0.y) * factor,
            f0.z + (1.0f - f0.z) * factor};
}

RT_HOST_DEVICE RT_FORCE_INLINE float fresnel_dielectric(float cosine_i,
                                                         float eta_i,
                                                         float eta_t) {
    cosine_i = saturate(cosine_i);
    const float sine_i =
        ::sqrtf(math::maximum(0.0f, 1.0f - cosine_i * cosine_i));
    const float sine_t = eta_i / eta_t * sine_i;
    if (sine_t >= 1.0f) {
        return 1.0f;
    }
    const float cosine_t =
        ::sqrtf(math::maximum(0.0f, 1.0f - sine_t * sine_t));
    const float parallel_numerator =
        eta_t * cosine_i - eta_i * cosine_t;
    const float parallel_denominator =
        eta_t * cosine_i + eta_i * cosine_t;
    const float perpendicular_numerator =
        eta_i * cosine_i - eta_t * cosine_t;
    const float perpendicular_denominator =
        eta_i * cosine_i + eta_t * cosine_t;
    if (math::absolute(parallel_denominator) <= 1e-12f ||
        math::absolute(perpendicular_denominator) <= 1e-12f) {
        return 1.0f;
    }
    const float parallel = parallel_numerator / parallel_denominator;
    const float perpendicular =
        perpendicular_numerator / perpendicular_denominator;
    return 0.5f *
           (parallel * parallel + perpendicular * perpendicular);
}

RT_HOST_DEVICE RT_FORCE_INLINE float ggx_alpha(float roughness) {
    return math::maximum(roughness * roughness, 1e-4f);
}

RT_HOST_DEVICE RT_FORCE_INLINE float distribution_ggx(float n_dot_h,
                                                       float alpha) {
    const float alpha2 = alpha * alpha;
    const float cosine2 = n_dot_h * n_dot_h;
    const float denominator = cosine2 * (alpha2 - 1.0f) + 1.0f;
    return alpha2 /
           math::maximum(kPi * denominator * denominator, 1e-12f);
}

RT_HOST_DEVICE RT_FORCE_INLINE float smith_g1(float n_dot_v, float alpha) {
    if (n_dot_v <= 0.0f) {
        return 0.0f;
    }
    const float cosine2 = n_dot_v * n_dot_v;
    const float tangent2 =
        math::maximum(0.0f, 1.0f - cosine2) /
        math::maximum(cosine2, 1e-12f);
    return 2.0f /
           (1.0f + ::sqrtf(1.0f + alpha * alpha * tangent2));
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 sample_visible_ggx_normal(
    Float3 local_wo, float alpha, float u0, float u1) {
    const Float3 stretched = math::normalize(
        {alpha * local_wo.x, alpha * local_wo.y, local_wo.z});
    const float length_xy_squared =
        stretched.x * stretched.x + stretched.y * stretched.y;
    const Float3 tangent1 =
        length_xy_squared > 1e-20f
            ? math::multiply({-stretched.y, stretched.x, 0.0f},
                             1.0f / ::sqrtf(length_xy_squared))
            : Float3{1.0f, 0.0f, 0.0f};
    const Float3 tangent2 = math::cross(stretched, tangent1);

    const float radius = ::sqrtf(saturate(u0));
    const float phi = 2.0f * kPi * saturate(u1);
    const float disk_x = radius * ::cosf(phi);
    float disk_y = radius * ::sinf(phi);
    const float blend = 0.5f * (1.0f + stretched.z);
    disk_y = (1.0f - blend) *
                 ::sqrtf(math::maximum(0.0f, 1.0f - disk_x * disk_x)) +
             blend * disk_y;
    const float disk_z = ::sqrtf(math::maximum(
        0.0f, 1.0f - disk_x * disk_x - disk_y * disk_y));
    const Float3 visible = math::add(
        math::multiply(tangent1, disk_x),
        math::add(math::multiply(tangent2, disk_y),
                  math::multiply(stretched, disk_z)));
    return math::normalize({alpha * visible.x, alpha * visible.y,
                            math::maximum(visible.z, 1e-8f)});
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 random_cosine_direction(RNG &rng) {
    const float u0 = static_cast<float>(rng.next());
    const float u1 = static_cast<float>(rng.next());
    const float z = ::sqrtf(math::maximum(0.0f, 1.0f - u1));
    const float phi = 2.0f * kPi * u0;
    const float radius = ::sqrtf(u1);
    return {::cosf(phi) * radius, ::sinf(phi) * radius, z};
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 random_unit_vector(RNG &rng) {
    for (;;) {
        const Float3 candidate{
            2.0f * static_cast<float>(rng.next()) - 1.0f,
            2.0f * static_cast<float>(rng.next()) - 1.0f,
            2.0f * static_cast<float>(rng.next()) - 1.0f};
        const float length_squared = math::length_squared(candidate);
        if (length_squared > 0.0f && length_squared < 1.0f) {
            return math::multiply(candidate, 1.0f / ::sqrtf(length_squared));
        }
    }
}

RT_HOST_DEVICE RT_FORCE_INLINE Float3 closure_eval(
    const PackedClosure &closure, const PackedShadingFrame &frame,
    Float3 wo, Float3 wi) {
    const Float3 parameters{closure.parameters.x, closure.parameters.y,
                            closure.parameters.z};
    switch (closure.type) {
    case PackedClosureType::Lambertian:
        if (math::dot(frame.normal, wo) <= 0.0f ||
            math::dot(frame.normal, wi) <= 0.0f) {
            return {};
        }
        return math::multiply(parameters, kInvPi);
    case PackedClosureType::GGXReflection:
    case PackedClosureType::ClearcoatGGX: {
        const float n_dot_o = math::dot(frame.normal, wo);
        const float n_dot_i = math::dot(frame.normal, wi);
        const Float3 sum = math::add(wo, wi);
        if (n_dot_o <= 0.0f || n_dot_i <= 0.0f || math::near_zero(sum)) {
            return {};
        }
        const Float3 half_vector = math::normalize(sum);
        const float n_dot_h =
            math::maximum(math::dot(frame.normal, half_vector), 0.0f);
        const float o_dot_h =
            math::maximum(math::dot(wo, half_vector), 0.0f);
        const float alpha = ggx_alpha(closure.parameters.w);
        const float distribution = distribution_ggx(n_dot_h, alpha);
        const float geometry = smith_g1(n_dot_o, alpha) *
                               smith_g1(n_dot_i, alpha);
        const float scale = distribution * geometry /
                            math::maximum(4.0f * n_dot_o * n_dot_i,
                                          1e-12f);
        return math::multiply(fresnel_schlick(o_dot_h, parameters), scale);
    }
    case PackedClosureType::IsotropicPhase:
        return math::multiply(parameters, kInvFourPi);
    case PackedClosureType::Mirror:
    case PackedClosureType::Dielectric:
        return {};
    }
    return {};
}

RT_HOST_DEVICE RT_FORCE_INLINE float closure_pdf(
    const PackedClosure &closure, const PackedShadingFrame &frame,
    Float3 wo, Float3 wi) {
    switch (closure.type) {
    case PackedClosureType::Lambertian:
        if (math::dot(frame.normal, wo) <= 0.0f ||
            math::dot(frame.normal, wi) <= 0.0f) {
            return 0.0f;
        }
        return math::dot(frame.normal, wi) * kInvPi;
    case PackedClosureType::GGXReflection:
    case PackedClosureType::ClearcoatGGX: {
        const float n_dot_o = math::dot(frame.normal, wo);
        const Float3 sum = math::add(wo, wi);
        if (n_dot_o <= 0.0f || math::dot(frame.normal, wi) <= 0.0f ||
            math::near_zero(sum)) {
            return 0.0f;
        }
        const Float3 half_vector = math::normalize(sum);
        if (math::dot(wo, half_vector) <= 0.0f) {
            return 0.0f;
        }
        const float n_dot_h =
            math::maximum(math::dot(frame.normal, half_vector), 0.0f);
        const float alpha = ggx_alpha(closure.parameters.w);
        const float result =
            distribution_ggx(n_dot_h, alpha) * smith_g1(n_dot_o, alpha) /
            math::maximum(4.0f * n_dot_o, 1e-12f);
        return math::finite(result) ? result : 0.0f;
    }
    case PackedClosureType::IsotropicPhase:
        return kInvFourPi;
    case PackedClosureType::Mirror:
    case PackedClosureType::Dielectric:
        return 0.0f;
    }
    return 0.0f;
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedBSDFStatus eval_packed_bsdf_core(
    const PackedMaterialOutput &output, Float3 wo, Float3 wi,
    Float3 &result) {
    result = {};
    if (output.closure_count > PackedMaterialOutput::kMaxClosures ||
        !math::finite(wo) || !math::finite(wi) ||
        !math::finite(output.geometry_normal)) {
        return PackedBSDFStatus::InvalidInput;
    }
    if (output.closure_count == 0) {
        return PackedBSDFStatus::Empty;
    }
    wo = math::normalize(wo);
    wi = math::normalize(wi);
    if (math::near_zero(wo) || math::near_zero(wi)) {
        return PackedBSDFStatus::InvalidInput;
    }
    if (!output_is_phase(output) &&
        !valid_event(output, wo, wi,
                     PACKED_BSDF_REFLECTION | PACKED_BSDF_GLOSSY)) {
        return PackedBSDFStatus::Success;
    }
    for (std::uint32_t index = 0; index < output.closure_count; ++index) {
        const PackedClosure &closure = output.closures[index];
        result = math::add(
            result,
            math::multiply(closure_eval(closure, output.frame, wo, wi),
                           closure.contribution_weight));
    }
    return math::finite(result) ? PackedBSDFStatus::Success
                                : PackedBSDFStatus::NonFinite;
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedBSDFStatus pdf_packed_bsdf_core(
    const PackedMaterialOutput &output, Float3 wo, Float3 wi,
    float &result) {
    result = 0.0f;
    if (output.closure_count > PackedMaterialOutput::kMaxClosures ||
        !math::finite(wo) || !math::finite(wi) ||
        !math::finite(output.geometry_normal)) {
        return PackedBSDFStatus::InvalidInput;
    }
    if (output.closure_count == 0) {
        return PackedBSDFStatus::Empty;
    }
    wo = math::normalize(wo);
    wi = math::normalize(wi);
    if (math::near_zero(wo) || math::near_zero(wi)) {
        return PackedBSDFStatus::InvalidInput;
    }
    const float total_weight = total_sample_weight(output);
    if (!(total_weight > 0.0f) || !math::finite(total_weight)) {
        return PackedBSDFStatus::InvalidInput;
    }
    if (!output_is_phase(output) &&
        !valid_event(output, wo, wi,
                     PACKED_BSDF_REFLECTION | PACKED_BSDF_GLOSSY)) {
        return PackedBSDFStatus::Success;
    }
    for (std::uint32_t index = 0; index < output.closure_count; ++index) {
        const PackedClosure &closure = output.closures[index];
        const float selection_pdf =
            math::maximum(closure.sample_weight, 0.0f) / total_weight;
        result += selection_pdf *
                  closure_pdf(closure, output.frame, wo, wi);
    }
    if (!math::finite(result) || result < 0.0f) {
        result = 0.0f;
        return PackedBSDFStatus::NonFinite;
    }
    return PackedBSDFStatus::Success;
}

// Distinct fast variants used only by the host fast transport path.
// Callers guarantee finite, normalized unit directions.
RT_HOST_DEVICE RT_FORCE_INLINE PackedBSDFStatus
eval_packed_bsdf_core_fast(const PackedMaterialOutput &output, Float3 wo,
                           Float3 wi, Float3 &result) {
    result = {};
    if (output.closure_count == 0) {
        return PackedBSDFStatus::Empty;
    }
    if (!output_is_phase(output) &&
        !valid_event(output, wo, wi,
                     PACKED_BSDF_REFLECTION | PACKED_BSDF_GLOSSY)) {
        return PackedBSDFStatus::Success;
    }
    for (std::uint32_t index = 0; index < output.closure_count; ++index) {
        const PackedClosure &closure = output.closures[index];
        result = math::add(
            result,
            math::multiply(closure_eval(closure, output.frame, wo, wi),
                           closure.contribution_weight));
    }
    return math::finite(result) ? PackedBSDFStatus::Success
                                : PackedBSDFStatus::NonFinite;
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedBSDFStatus
pdf_packed_bsdf_core_fast(const PackedMaterialOutput &output, Float3 wo,
                          Float3 wi, float &result) {
    result = 0.0f;
    if (output.closure_count == 0) {
        return PackedBSDFStatus::Empty;
    }
    const float total_weight = total_sample_weight(output);
    if (!(total_weight > 0.0f)) {
        return PackedBSDFStatus::InvalidInput;
    }
    if (!output_is_phase(output) &&
        !valid_event(output, wo, wi,
                     PACKED_BSDF_REFLECTION | PACKED_BSDF_GLOSSY)) {
        return PackedBSDFStatus::Success;
    }
    for (std::uint32_t index = 0; index < output.closure_count; ++index) {
        const PackedClosure &closure = output.closures[index];
        const float selection_pdf =
            math::maximum(closure.sample_weight, 0.0f) / total_weight;
        result += selection_pdf *
                  closure_pdf(closure, output.frame, wo, wi);
    }
    if (!math::finite(result) || result < 0.0f) {
        result = 0.0f;
        return PackedBSDFStatus::NonFinite;
    }
    return PackedBSDFStatus::Success;
}

RT_HOST_DEVICE RT_FORCE_INLINE float abs_cos_theta_fast(
    const PackedMaterialOutput &output, Float3 direction) {
    if (output_is_phase(output)) {
        return 1.0f;
    }
    return math::absolute(math::dot(direction, output.frame.normal));
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedBSDFStatus sample_closure(
    const PackedClosure &closure, const PackedShadingFrame &frame,
    Float3 wo, RNG &rng, PackedBSDFSample &sample) {
    const Float3 parameters{closure.parameters.x, closure.parameters.y,
                            closure.parameters.z};
    switch (closure.type) {
    case PackedClosureType::Lambertian:
        if (math::dot(frame.normal, wo) <= 0.0f) {
            return PackedBSDFStatus::NoSample;
        }
        sample.wi = math::normalize(
            frame_to_world(frame,
                           packed_bsdf::random_cosine_direction(rng)));
        sample.f = math::multiply(parameters, kInvPi);
        sample.pdf =
            math::maximum(math::dot(frame.normal, sample.wi), 0.0f) *
            kInvPi;
        sample.flags = PACKED_BSDF_DIFFUSE | PACKED_BSDF_REFLECTION;
        break;
    case PackedClosureType::Mirror: {
        if (math::dot(frame.normal, wo) <= 0.0f) {
            return PackedBSDFStatus::NoSample;
        }
        sample.wi = math::normalize(
            math::reflect(math::multiply(wo, -1.0f), frame.normal));
        const float cosine =
            math::absolute(math::dot(frame.normal, sample.wi));
        if (!(cosine > 0.0f)) {
            return PackedBSDFStatus::NoSample;
        }
        sample.f = math::multiply(parameters, 1.0f / cosine);
        sample.pdf = 1.0f;
        sample.flags = PACKED_BSDF_DELTA | PACKED_BSDF_REFLECTION;
        break;
    }
    case PackedClosureType::Dielectric: {
        const float cosine_o = math::dot(frame.normal, wo);
        if (cosine_o <= 0.0f) {
            return PackedBSDFStatus::NoSample;
        }
        const bool front_face =
            (closure.flags & PACKED_CLOSURE_FRONT_FACE) != 0;
        const float ior = math::maximum(closure.parameters.x, 1.0001f);
        const float eta_i = front_face ? 1.0f : ior;
        const float eta_t = front_face ? ior : 1.0f;
        const float eta_ratio = eta_i / eta_t;
        const float reflectance =
            fresnel_dielectric(cosine_o, eta_i, eta_t);
        if (static_cast<float>(rng.next()) < reflectance) {
            sample.wi = math::normalize(
                math::reflect(math::multiply(wo, -1.0f), frame.normal));
            const float cosine = math::absolute(
                math::dot(frame.normal, sample.wi));
            sample.f = {reflectance / math::maximum(cosine, 1e-12f),
                        reflectance / math::maximum(cosine, 1e-12f),
                        reflectance / math::maximum(cosine, 1e-12f)};
            sample.pdf = reflectance;
            sample.flags = PACKED_BSDF_DELTA | PACKED_BSDF_REFLECTION;
        } else {
            sample.wi = math::normalize(math::refract(
                math::multiply(wo, -1.0f), frame.normal, eta_ratio));
            const float cosine = math::absolute(
                math::dot(frame.normal, sample.wi));
            const float transmittance = 1.0f - reflectance;
            const float radiance_scale = eta_ratio * eta_ratio;
            const float value = transmittance * radiance_scale /
                                math::maximum(cosine, 1e-12f);
            sample.f = {value, value, value};
            sample.pdf = transmittance;
            sample.flags = PACKED_BSDF_DELTA | PACKED_BSDF_TRANSMISSION;
            sample.eta = eta_t / eta_i;
        }
        break;
    }
    case PackedClosureType::GGXReflection:
    case PackedClosureType::ClearcoatGGX: {
        const Float3 local_wo = frame_to_local(frame, wo);
        if (local_wo.z <= 0.0f || !math::finite(local_wo)) {
            return PackedBSDFStatus::NoSample;
        }
        const float alpha = ggx_alpha(closure.parameters.w);
        const float u0 = static_cast<float>(rng.next());
        const float u1 = static_cast<float>(rng.next());
        const Float3 local_half = sample_visible_ggx_normal(
            math::normalize(local_wo), alpha, u0, u1);
        const Float3 half_vector =
            math::normalize(frame_to_world(frame, local_half));
        sample.wi = math::normalize(
            math::reflect(math::multiply(wo, -1.0f), half_vector));
        if (!math::finite(sample.wi) ||
            math::dot(frame.normal, sample.wi) <= 0.0f) {
            return PackedBSDFStatus::NoSample;
        }
        sample.f = closure_eval(closure, frame, wo, sample.wi);
        sample.pdf = closure_pdf(closure, frame, wo, sample.wi);
        sample.flags = PACKED_BSDF_GLOSSY | PACKED_BSDF_REFLECTION;
        break;
    }
    case PackedClosureType::IsotropicPhase:
        sample.wi = packed_bsdf::random_unit_vector(rng);
        sample.f = math::multiply(parameters, kInvFourPi);
        sample.pdf = kInvFourPi;
        sample.flags = PACKED_BSDF_PHASE;
        break;
    }
    if (!(sample.pdf > 0.0f)) {
        return PackedBSDFStatus::NoSample;
    }
    return math::finite(sample.wi) && math::finite(sample.f) &&
                   math::finite(sample.pdf) && math::finite(sample.eta)
               ? PackedBSDFStatus::Success
               : PackedBSDFStatus::NonFinite;
}

template <bool kFastBsdf>
RT_HOST_DEVICE RT_FORCE_INLINE PackedBSDFStatus
sample_packed_bsdf_core_impl(const PackedMaterialOutput &output, Float3 wo,
                             RNG &rng, PackedBSDFSample &sample) {
    sample = {};
    if (output.closure_count > PackedMaterialOutput::kMaxClosures ||
        !math::finite(wo) || !math::finite(output.geometry_normal)) {
        return PackedBSDFStatus::InvalidInput;
    }
    if (output.closure_count == 0) {
        return PackedBSDFStatus::Empty;
    }
    wo = math::normalize(wo);
    if (math::near_zero(wo)) {
        return PackedBSDFStatus::InvalidInput;
    }
    const float total_weight = total_sample_weight(output);
    if (!(total_weight > 0.0f) || !math::finite(total_weight)) {
        return PackedBSDFStatus::InvalidInput;
    }
    const float target = static_cast<float>(rng.next()) * total_weight;
    float accumulated = 0.0f;
    std::uint32_t selected = output.closure_count - 1;
    for (std::uint32_t index = 0; index < output.closure_count; ++index) {
        accumulated +=
            math::maximum(output.closures[index].sample_weight, 0.0f);
        if (target < accumulated) {
            selected = index;
            break;
        }
    }
    const PackedClosure &closure = output.closures[selected];
    const float selection_pdf =
        math::maximum(closure.sample_weight, 0.0f) / total_weight;
    sample.closure_index = selected;
    PackedBSDFStatus status =
        sample_closure(closure, output.frame, wo, rng, sample);
    if (status != PackedBSDFStatus::Success) {
        return status;
    }
    if (!valid_event(output, wo, sample.wi, sample.flags)) {
        sample = {};
        return PackedBSDFStatus::NoSample;
    }
    sample.closure_index = selected;
    if (sample.is_delta()) {
        sample.f =
            math::multiply(sample.f, closure.contribution_weight);
        sample.pdf *= selection_pdf;
    } else if constexpr (kFastBsdf) {
        status = eval_packed_bsdf_core_fast(output, wo, sample.wi, sample.f);
        if (status != PackedBSDFStatus::Success) {
            sample = {};
            return status;
        }
        status = pdf_packed_bsdf_core_fast(output, wo, sample.wi, sample.pdf);
        if (status != PackedBSDFStatus::Success || !(sample.pdf > 0.0f)) {
            sample = {};
            return status == PackedBSDFStatus::Success
                       ? PackedBSDFStatus::NoSample
                       : status;
        }
    } else {
        status = eval_packed_bsdf_core(output, wo, sample.wi, sample.f);
        if (status != PackedBSDFStatus::Success) {
            sample = {};
            return status;
        }
        status = pdf_packed_bsdf_core(output, wo, sample.wi, sample.pdf);
        if (status != PackedBSDFStatus::Success || !(sample.pdf > 0.0f)) {
            sample = {};
            return status == PackedBSDFStatus::Success
                       ? PackedBSDFStatus::NoSample
                       : status;
        }
    }
    if (!math::finite(sample.wi) || !math::finite(sample.f) ||
        !math::finite(sample.pdf) || !math::finite(sample.eta)) {
        sample = {};
        return PackedBSDFStatus::NonFinite;
    }
    return PackedBSDFStatus::Success;
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedBSDFStatus sample_packed_bsdf_core(
    const PackedMaterialOutput &output, Float3 wo, RNG &rng,
    PackedBSDFSample &sample) {
    return sample_packed_bsdf_core_impl<false>(output, wo, rng, sample);
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedBSDFStatus
sample_packed_bsdf_core_fast(const PackedMaterialOutput &output, Float3 wo,
                             RNG &rng, PackedBSDFSample &sample) {
    return sample_packed_bsdf_core_impl<true>(output, wo, rng, sample);
}

RT_HOST_DEVICE RT_FORCE_INLINE float abs_cos_theta(
    const PackedMaterialOutput &output, Float3 direction) {
    if (output_is_phase(output)) {
        return 1.0f;
    }
    const Float3 unit_direction = math::normalize(direction);
    return math::absolute(math::dot(unit_direction, output.frame.normal));
}

} // namespace packed_bsdf

#endif

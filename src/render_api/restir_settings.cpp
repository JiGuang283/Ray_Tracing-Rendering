#include "restir_settings.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace {

void hash_u32(std::uint64_t &hash, std::uint32_t value) noexcept {
    constexpr std::uint64_t prime = 1099511628211ull;
    for (std::uint32_t byte = 0; byte < 4; ++byte) {
        hash ^= (value >> (byte * 8u)) & 0xffu;
        hash *= prime;
    }
}

std::uint32_t float_bits(float value) noexcept {
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

} // namespace

void validate_restir_settings(const RestirSettings &settings) {
    if (settings.initial_light_candidates == 0u &&
        settings.initial_bsdf_candidates == 0u) {
        throw std::invalid_argument(
            "ReSTIR requires at least one initial candidate");
    }
    if (settings.spatial_reuse &&
        (settings.spatial_neighbors == 0u ||
         settings.spatial_passes == 0u)) {
        throw std::invalid_argument(
            "ReSTIR spatial reuse requires neighbors and passes");
    }
    if (settings.spatial_neighbors > 64u) {
        throw std::invalid_argument(
            "ReSTIR spatial reuse supports at most 64 unique neighbors");
    }
    if (settings.temporal_reuse && settings.max_history_length == 0u) {
        throw std::invalid_argument(
            "ReSTIR temporal reuse requires nonzero history length");
    }
    if (!std::isfinite(settings.normal_threshold) ||
        settings.normal_threshold < -1.0f ||
        settings.normal_threshold > 1.0f) {
        throw std::invalid_argument(
            "ReSTIR normal threshold must be finite and in [-1, 1]");
    }
    if (!std::isfinite(settings.depth_threshold) ||
        settings.depth_threshold < 0.0f) {
        throw std::invalid_argument(
            "ReSTIR depth threshold must be finite and non-negative");
    }
    if (!std::isfinite(settings.roughness_reconnect_threshold) ||
        settings.roughness_reconnect_threshold < 0.0f ||
        settings.roughness_reconnect_threshold > 1.0f) {
        throw std::invalid_argument(
            "ReSTIR reconnect threshold must be finite and in [0, 1]");
    }
    if (settings.bias_correction == RestirBiasCorrection::Unbiased &&
        settings.max_reservoir_candidates != 0u) {
        throw std::invalid_argument(
            "unbiased ReSTIR cannot use an uncorrected reservoir M cap");
    }
}

std::uint64_t restir_settings_fingerprint(
    const RestirSettings &settings) noexcept {
    std::uint64_t hash = 1469598103934665603ull;
    hash_u32(hash, settings.initial_light_candidates);
    hash_u32(hash, settings.initial_bsdf_candidates);
    hash_u32(hash, settings.spatial_neighbors);
    hash_u32(hash, settings.spatial_passes);
    hash_u32(hash, settings.max_history_length);
    hash_u32(hash, settings.max_reservoir_candidates);
    hash_u32(hash, float_bits(settings.normal_threshold));
    hash_u32(hash, float_bits(settings.depth_threshold));
    hash_u32(hash, float_bits(settings.roughness_reconnect_threshold));
    hash_u32(hash, settings.temporal_reuse ? 1u : 0u);
    hash_u32(hash, settings.spatial_reuse ? 1u : 0u);
    hash_u32(hash, settings.visibility_reuse ? 1u : 0u);
    hash_u32(hash, static_cast<std::uint32_t>(settings.bias_correction));
    return hash;
}

#ifndef RESTIR_SETTINGS_H
#define RESTIR_SETTINGS_H

#include <array>
#include <cstdint>

enum class RestirBiasCorrection : std::uint32_t {
    Unbiased = 0,
    Basic = 1,
    Pairwise = 2,
};

enum class RestirHistoryMode : std::uint32_t {
    Reset = 0,
    Continue = 1,
    Auto = 2,
};

struct RestirSettings {
    std::uint32_t initial_light_candidates = 8;
    std::uint32_t initial_bsdf_candidates = 1;
    std::uint32_t spatial_neighbors = 5;
    std::uint32_t spatial_passes = 1;
    std::uint32_t max_history_length = 20;
    // Zero means unlimited. Unbiased mode rejects a finite cap.
    std::uint32_t max_reservoir_candidates = 32;
    float normal_threshold = 0.9f;
    float depth_threshold = 0.1f;
    float roughness_reconnect_threshold = 0.2f;
    bool temporal_reuse = true;
    bool spatial_reuse = true;
    bool visibility_reuse = false;
    RestirBiasCorrection bias_correction =
        RestirBiasCorrection::Pairwise;
    RestirHistoryMode history_mode = RestirHistoryMode::Auto;
};

constexpr std::size_t kRestirShiftFailureBuckets = 16;
constexpr std::size_t kRestirHistoryFailureBuckets = 16;

struct RestirStats {
    std::uint64_t iterations = 0;
    std::uint64_t initial_candidates = 0;
    std::uint64_t temporal_candidates = 0;
    std::uint64_t temporal_accepted = 0;
    std::uint64_t spatial_candidates = 0;
    std::uint64_t spatial_accepted = 0;
    std::uint64_t visibility_rays = 0;
    std::uint64_t history_resets = 0;
    std::uint64_t invalid_reservoirs = 0;
    std::uint64_t shift_success = 0;
    std::array<std::uint64_t, kRestirShiftFailureBuckets> shift_failures{};
    std::array<std::uint64_t, kRestirHistoryFailureBuckets>
        history_failures{};
    double average_M = 0.0;
    double average_age = 0.0;
};

void validate_restir_settings(const RestirSettings &settings);
std::uint64_t restir_settings_fingerprint(
    const RestirSettings &settings) noexcept;

#endif

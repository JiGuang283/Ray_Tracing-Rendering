#include "restir_reference.h"

#include "reservoir_core.h"
#include "rng.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace restir {
namespace {

using ReferenceReservoir = Reservoir<std::uint32_t, double>;

void require_success(const ReservoirOperationResult &result) {
    if (!result.accepted()) {
        throw std::runtime_error("reference reservoir operation failed");
    }
}

std::uint32_t select_weighted(const std::array<double, 4> &weights,
                              std::uint32_t count, RNG &rng,
                              bool reverse_order = false) {
    ReferenceReservoir reservoir;
    for (std::uint32_t step = 0; step < count; ++step) {
        const std::uint32_t index =
            reverse_order ? count - step - 1u : step;
        require_success(stream_candidate(
            reservoir,
            make_candidate(index, weights[index], 1.0, true, true),
            rng.next()));
    }
    require_success(finalize_reservoir(reservoir));
    return reservoir.sample;
}

double confidence_tolerance(double probability, std::uint64_t trials) {
    const double variance =
        probability * (1.0 - probability) / static_cast<double>(trials);
    return 6.0 * std::sqrt(variance) + 0.001;
}

} // namespace

bool RestirReferenceReport::passed() const noexcept {
    return trials > 0 &&
           std::abs(heavy_observed_probability - heavy_expected_probability) <=
               heavy_tolerance &&
           uniform_max_error <= confidence_tolerance(0.25, trials) &&
           std::abs(constant_mean_estimate - constant_expected_estimate) <=
               1e-12 &&
           forward_reverse_max_difference <=
               2.0 * confidence_tolerance(0.125, trials);
}

RestirReferenceReport run_restir_reference_validation(std::uint64_t trials,
                                                      std::uint32_t seed) {
    if (trials == 0) {
        throw std::invalid_argument("ReSTIR reference trials must be positive");
    }

    RestirReferenceReport report;
    report.trials = trials;

    RNG heavy_rng(mix_seed(seed, 1u));
    std::uint64_t heavy_count = 0;
    for (std::uint64_t trial = 0; trial < trials; ++trial) {
        const std::array<double, 4> weights{1.0, 3.0, 0.0, 0.0};
        heavy_count += select_weighted(weights, 2u, heavy_rng) == 1u ? 1u : 0u;
    }
    report.heavy_expected_probability = 0.75;
    report.heavy_observed_probability =
        static_cast<double>(heavy_count) / static_cast<double>(trials);
    report.heavy_tolerance = confidence_tolerance(0.75, trials);

    RNG uniform_rng(mix_seed(seed, 2u));
    std::array<std::uint64_t, 4> uniform_counts{};
    for (std::uint64_t trial = 0; trial < trials; ++trial) {
        const std::array<double, 4> weights{1.0, 1.0, 1.0, 1.0};
        ++uniform_counts[select_weighted(weights, 4u, uniform_rng)];
    }
    for (std::size_t index = 0; index < uniform_counts.size(); ++index) {
        report.uniform_observed_probabilities[index] =
            static_cast<double>(uniform_counts[index]) /
            static_cast<double>(trials);
        report.uniform_max_error =
            std::max(report.uniform_max_error,
                     std::abs(report.uniform_observed_probabilities[index] -
                              0.25));
    }

    RNG constant_rng(mix_seed(seed, 3u));
    double constant_sum = 0.0;
    constexpr std::uint32_t candidates_per_trial = 8u;
    for (std::uint64_t trial = 0; trial < trials; ++trial) {
        ReferenceReservoir reservoir;
        for (std::uint32_t candidate_index = 0;
             candidate_index < candidates_per_trial; ++candidate_index) {
            const std::uint32_t sample = constant_rng.next() < 0.5 ? 0u : 1u;
            require_success(stream_candidate(
                reservoir, make_candidate(sample, 2.0, 0.5, true, true),
                constant_rng.next()));
        }
        require_success(finalize_reservoir(reservoir));
        constant_sum += 2.0 * reservoir.unbiased_contribution_weight;
    }
    report.constant_expected_estimate = 4.0;
    report.constant_mean_estimate =
        constant_sum / static_cast<double>(trials);

    RNG forward_rng(mix_seed(seed, 4u));
    RNG reverse_rng(mix_seed(seed, 5u));
    std::array<std::uint64_t, 4> forward_counts{};
    std::array<std::uint64_t, 4> reverse_counts{};
    const std::array<double, 4> ordered_weights{1.0, 2.0, 5.0, 0.0};
    for (std::uint64_t trial = 0; trial < trials; ++trial) {
        ++forward_counts[select_weighted(ordered_weights, 3u, forward_rng)];
        ++reverse_counts[
            select_weighted(ordered_weights, 3u, reverse_rng, true)];
    }
    for (std::uint32_t index = 0; index < 3u; ++index) {
        const double forward = static_cast<double>(forward_counts[index]) /
                               static_cast<double>(trials);
        const double reverse = static_cast<double>(reverse_counts[index]) /
                               static_cast<double>(trials);
        report.forward_reverse_max_difference =
            std::max(report.forward_reverse_max_difference,
                     std::abs(forward - reverse));
    }
    return report;
}

} // namespace restir

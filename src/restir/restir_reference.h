#ifndef RESTIR_REFERENCE_H
#define RESTIR_REFERENCE_H

#include <array>
#include <cstdint>

namespace restir {

struct RestirReferenceReport {
    std::uint64_t trials = 0;
    double heavy_expected_probability = 0.0;
    double heavy_observed_probability = 0.0;
    double heavy_tolerance = 0.0;
    std::array<double, 4> uniform_observed_probabilities{};
    double uniform_max_error = 0.0;
    double constant_expected_estimate = 0.0;
    double constant_mean_estimate = 0.0;
    double forward_reverse_max_difference = 0.0;

    bool passed() const noexcept;
};

RestirReferenceReport run_restir_reference_validation(
    std::uint64_t trials = 200000, std::uint32_t seed = 1337);

} // namespace restir

#endif

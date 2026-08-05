#include "test_harness.h"

#include "gris_core.h"
#include "reservoir_core.h"
#include "restir_reference.h"
#include "restir_validation.h"
#include "rng.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>

namespace {

using DoubleReservoir = restir::Reservoir<std::uint32_t, double>;

void require_accepted(const restir::ReservoirOperationResult &result) {
    REQUIRE(result.accepted());
}

DoubleReservoir make_source(std::uint32_t sample, double target,
                            double proposal_pdf) {
    DoubleReservoir reservoir;
    require_accepted(restir::stream_candidate(
        reservoir,
        restir::make_candidate(sample, target, proposal_pdf, true, true),
        0.25));
    require_accepted(restir::finalize_reservoir(reservoir));
    return reservoir;
}

double binomial_tolerance(double probability, std::uint64_t trials) {
    return 6.0 *
               std::sqrt(probability * (1.0 - probability) /
                         static_cast<double>(trials)) +
           0.001;
}

} // namespace

TEST_CASE(restir_reservoir_pod_layout_is_stable) {
    using FloatReservoir = restir::Reservoir<std::uint32_t, float>;
    using FloatCandidate =
        restir::ReservoirCandidate<std::uint32_t, float>;
    REQUIRE(std::is_trivially_copyable_v<FloatReservoir>);
    REQUIRE(std::is_trivially_copyable_v<FloatCandidate>);
    REQUIRE(std::is_trivially_copyable_v<restir::RestirShiftResult<float>>);
    REQUIRE(sizeof(FloatReservoir) == 32u);
    REQUIRE(sizeof(DoubleReservoir) == 48u);
    REQUIRE(alignof(FloatReservoir) == 16u);
}

TEST_CASE(restir_streaming_update_and_finalize_keep_distinct_weights) {
    DoubleReservoir reservoir;
    const auto first = restir::stream_candidate(
        reservoir, restir::make_candidate(7u, 2.0, 0.5), 0.9);
    const auto second = restir::stream_candidate(
        reservoir, restir::make_candidate(9u, 6.0, 0.5), 0.1);
    require_accepted(first);
    require_accepted(second);
    REQUIRE(first.changed_selection());
    REQUIRE(second.changed_selection());
    REQUIRE(reservoir.sample == 9u);
    REQUIRE(reservoir.M == 2u);
    REQUIRE_NEAR(reservoir.weight_sum, 16.0, 1e-12);
    REQUIRE_NEAR(reservoir.selected_target, 6.0, 1e-12);
    REQUIRE_NEAR(reservoir.unbiased_contribution_weight, 0.0, 1e-12);

    require_accepted(restir::finalize_reservoir(reservoir));
    REQUIRE(restir::reservoir_is_usable(reservoir));
    REQUIRE_NEAR(reservoir.weight_sum, 16.0, 1e-12);
    REQUIRE_NEAR(reservoir.unbiased_contribution_weight, 4.0 / 3.0,
                 1e-12);

    const auto late_update = restir::stream_candidate(
        reservoir, restir::make_candidate(11u, 1.0, 1.0), 0.5);
    REQUIRE(late_update.rejection ==
            restir::ReservoirRejectReason::ReservoirFinalized);
    REQUIRE(reservoir.M == 2u);
}

TEST_CASE(restir_zero_target_counts_the_draw_without_selecting_it) {
    DoubleReservoir reservoir;
    const auto zero = restir::stream_candidate(
        reservoir,
        restir::make_candidate(1u, 0.0, 0.5, false, true), 0.5);
    REQUIRE(zero.rejection == restir::ReservoirRejectReason::ZeroTarget);
    REQUIRE(zero.represented_candidates == 1u);
    REQUIRE(reservoir.M == 1u);
    REQUIRE(!restir::reservoir_has_sample(reservoir));

    require_accepted(restir::stream_candidate(
        reservoir, restir::make_candidate(2u, 2.0, 1.0), 0.5));
    require_accepted(restir::finalize_reservoir(reservoir));
    REQUIRE(reservoir.M == 2u);
    REQUIRE_NEAR(reservoir.unbiased_contribution_weight, 0.5, 1e-12);
}

TEST_CASE(restir_rejects_invalid_candidates_without_hidden_mutation) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();
    DoubleReservoir reservoir;

    const auto invalid_sample = restir::stream_candidate(
        reservoir,
        restir::make_candidate(1u, 1.0, 1.0, true, false), 0.5);
    REQUIRE(invalid_sample.rejection ==
            restir::ReservoirRejectReason::InvalidSample);
    REQUIRE(reservoir.M == 0u);

    const auto negative_target = restir::stream_candidate(
        reservoir, restir::make_candidate(1u, -1.0, 1.0), 0.5);
    REQUIRE(negative_target.rejection ==
            restir::ReservoirRejectReason::NegativeTarget);
    const auto infinite_target = restir::stream_candidate(
        reservoir, restir::make_candidate(1u, infinity, 1.0), 0.5);
    REQUIRE(infinite_target.rejection ==
            restir::ReservoirRejectReason::NonFiniteTarget);
    const auto zero_pdf = restir::stream_candidate(
        reservoir, restir::make_candidate(1u, 1.0, 0.0), 0.5);
    REQUIRE(zero_pdf.rejection ==
            restir::ReservoirRejectReason::ZeroProposalWithContribution);
    const auto nan_pdf = restir::stream_candidate(
        reservoir, restir::make_candidate(1u, 1.0, nan), 0.5);
    REQUIRE(nan_pdf.rejection ==
            restir::ReservoirRejectReason::NonFiniteProposalPdf);
    const auto nan_random = restir::stream_candidate(
        reservoir, restir::make_candidate(1u, 1.0, 1.0), nan);
    REQUIRE(nan_random.rejection ==
            restir::ReservoirRejectReason::NonFiniteRandom);
    const auto out_of_range_random = restir::stream_candidate(
        reservoir, restir::make_candidate(1u, 1.0, 1.0), 1.0);
    REQUIRE(out_of_range_random.rejection ==
            restir::ReservoirRejectReason::RandomOutOfRange);
    const auto mismatched_zero = restir::stream_candidate(
        reservoir, restir::make_candidate(1u, 0.0, 1.0, true, true),
        0.5);
    REQUIRE(mismatched_zero.rejection ==
            restir::ReservoirRejectReason::ZeroTargetWithContribution);
    REQUIRE(reservoir.M == 0u);
    REQUIRE_NEAR(reservoir.weight_sum, 0.0, 1e-12);
}

TEST_CASE(restir_rejects_candidate_count_and_weight_overflow) {
    DoubleReservoir count_overflow;
    count_overflow.M = std::numeric_limits<std::uint32_t>::max();
    const auto count_result = restir::stream_candidate(
        count_overflow, restir::make_candidate(1u, 1.0, 1.0), 0.5);
    REQUIRE(count_result.rejection ==
            restir::ReservoirRejectReason::CandidateCountOverflow);
    REQUIRE(count_overflow.M == std::numeric_limits<std::uint32_t>::max());

    DoubleReservoir weight_overflow;
    weight_overflow.sample = 1u;
    weight_overflow.weight_sum = std::numeric_limits<double>::max();
    weight_overflow.selected_target = 1.0;
    weight_overflow.M = 1u;
    weight_overflow.flags = restir::RESERVOIR_FLAG_HAS_SAMPLE;
    const auto weight_result = restir::stream_candidate(
        weight_overflow,
        restir::make_candidate(2u, std::numeric_limits<double>::max(), 1.0),
        0.5);
    REQUIRE(weight_result.rejection ==
            restir::ReservoirRejectReason::NonFiniteWeightSum);
    REQUIRE(weight_overflow.M == 1u);
    REQUIRE(weight_overflow.sample == 1u);
}

TEST_CASE(restir_reference_selection_statistics_match_offline_weights) {
    const restir::RestirReferenceReport report =
        restir::run_restir_reference_validation(60000u, 123u);
    REQUIRE(report.passed());
    REQUIRE(std::abs(report.heavy_observed_probability - 0.75) <=
            report.heavy_tolerance);
    REQUIRE(report.uniform_max_error <= binomial_tolerance(0.25, 60000u));
    REQUIRE_NEAR(report.constant_mean_estimate, 4.0, 1e-12);
}

TEST_CASE(restir_streaming_distribution_matches_offline_weighted_selection) {
    constexpr std::uint64_t trials = 80000u;
    constexpr double total_weight = 8.0;
    const double expected[3]{1.0 / total_weight, 2.0 / total_weight,
                             5.0 / total_weight};
    std::uint64_t streaming_counts[3]{};
    std::uint64_t offline_counts[3]{};
    RNG streaming_rng(1001u);
    RNG offline_rng(1002u);
    for (std::uint64_t trial = 0; trial < trials; ++trial) {
        DoubleReservoir reservoir;
        require_accepted(restir::stream_candidate(
            reservoir, restir::make_candidate(0u, 1.0, 1.0),
            streaming_rng.next()));
        require_accepted(restir::stream_candidate(
            reservoir, restir::make_candidate(1u, 2.0, 1.0),
            streaming_rng.next()));
        require_accepted(restir::stream_candidate(
            reservoir, restir::make_candidate(2u, 5.0, 1.0),
            streaming_rng.next()));
        ++streaming_counts[reservoir.sample];

        const double draw = offline_rng.next() * total_weight;
        const std::uint32_t offline_sample =
            draw < 1.0 ? 0u : (draw < 3.0 ? 1u : 2u);
        ++offline_counts[offline_sample];
    }
    for (std::uint32_t index = 0; index < 3u; ++index) {
        const double streaming =
            static_cast<double>(streaming_counts[index]) /
            static_cast<double>(trials);
        const double offline = static_cast<double>(offline_counts[index]) /
                               static_cast<double>(trials);
        const double tolerance = binomial_tolerance(expected[index], trials);
        REQUIRE(std::abs(streaming - expected[index]) <= tolerance);
        REQUIRE(std::abs(offline - expected[index]) <= tolerance);
        REQUIRE(std::abs(streaming - offline) <= 2.0 * tolerance);
    }
}

TEST_CASE(restir_extreme_weight_selection_remains_stable) {
    constexpr std::uint64_t trials = 50000u;
    RNG rng(71u);
    std::uint64_t heavy_count = 0;
    for (std::uint64_t trial = 0; trial < trials; ++trial) {
        DoubleReservoir reservoir;
        require_accepted(restir::stream_candidate(
            reservoir, restir::make_candidate(0u, 1.0, 1.0), rng.next()));
        require_accepted(restir::stream_candidate(
            reservoir, restir::make_candidate(1u, 1e6, 1.0), rng.next()));
        require_accepted(restir::finalize_reservoir(reservoir));
        heavy_count += reservoir.sample == 1u ? 1u : 0u;
    }
    const double observed =
        static_cast<double>(heavy_count) / static_cast<double>(trials);
    const double expected = 1e6 / (1e6 + 1.0);
    REQUIRE(std::abs(observed - expected) <=
            binomial_tolerance(expected, trials));
}

TEST_CASE(restir_identity_gris_combine_reconstructs_ris_mass) {
    DoubleReservoir source_a;
    require_accepted(restir::stream_candidate(
        source_a, restir::make_candidate(1u, 2.0, 1.0), 0.8));
    require_accepted(restir::stream_candidate(
        source_a, restir::make_candidate(2u, 6.0, 1.0), 0.1));
    require_accepted(restir::finalize_reservoir(source_a));

    const DoubleReservoir source_b = make_source(3u, 4.0, 2.0);
    DoubleReservoir combined;
    require_accepted(restir::combine_reservoir(
        combined, source_a, source_a.sample,
        restir::identity_shift(source_a.selected_target), 0.6));
    require_accepted(restir::combine_reservoir(
        combined, source_b, source_b.sample,
        restir::identity_shift(source_b.selected_target), 0.2));
    REQUIRE(combined.M == 3u);
    REQUIRE_NEAR(combined.weight_sum, 10.0, 1e-12);
    require_accepted(restir::finalize_reservoir(combined));
    REQUIRE_NEAR(combined.unbiased_contribution_weight,
                 combined.weight_sum /
                     (3.0 * combined.selected_target),
                 1e-12);
}

TEST_CASE(restir_identity_gris_combine_preserves_selection_distribution) {
    constexpr std::uint64_t trials = 80000u;
    const double expected[3]{0.125, 0.25, 0.625};
    std::uint64_t counts[3]{};
    RNG rng(2001u);
    for (std::uint64_t trial = 0; trial < trials; ++trial) {
        DoubleReservoir source_a;
        require_accepted(restir::stream_candidate(
            source_a, restir::make_candidate(0u, 1.0, 1.0), rng.next()));
        require_accepted(restir::stream_candidate(
            source_a, restir::make_candidate(1u, 2.0, 1.0), rng.next()));
        require_accepted(restir::finalize_reservoir(source_a));
        const DoubleReservoir source_b = make_source(2u, 5.0, 1.0);

        DoubleReservoir combined;
        require_accepted(restir::combine_reservoir(
            combined, source_a, source_a.sample,
            restir::identity_shift(source_a.selected_target), rng.next()));
        require_accepted(restir::combine_reservoir(
            combined, source_b, source_b.sample,
            restir::identity_shift(source_b.selected_target), rng.next()));
        require_accepted(restir::finalize_reservoir(combined));
        ++counts[combined.sample];
    }
    for (std::uint32_t index = 0; index < 3u; ++index) {
        const double observed = static_cast<double>(counts[index]) /
                                static_cast<double>(trials);
        REQUIRE(std::abs(observed - expected[index]) <=
                binomial_tolerance(expected[index], trials));
    }
}

TEST_CASE(restir_gris_applies_destination_target_jacobian_and_mis_weight) {
    const DoubleReservoir source = make_source(5u, 2.0, 0.5);
    REQUIRE_NEAR(source.unbiased_contribution_weight, 2.0, 1e-12);

    restir::RestirShiftResult<double> shift =
        restir::identity_shift(3.0, 0.25);
    shift.jacobian = 2.0;
    DoubleReservoir destination;
    require_accepted(restir::combine_reservoir(
        destination, source, 8u, shift, 0.5));
    REQUIRE(destination.M == 1u);
    REQUIRE(destination.sample == 8u);
    REQUIRE_NEAR(destination.weight_sum, 3.0, 1e-12);
    require_accepted(restir::finalize_reservoir(destination));
    REQUIRE_NEAR(destination.unbiased_contribution_weight, 1.0, 1e-12);

    double balance_weight = 0.0;
    REQUIRE(restir::compute_balance_mis_weight(2.0, 8.0,
                                               balance_weight));
    REQUIRE_NEAR(balance_weight, 0.25, 1e-12);
    REQUIRE(!restir::compute_balance_mis_weight(9.0, 8.0,
                                                balance_weight));
}

TEST_CASE(restir_gris_rejects_invalid_shifts_and_preserves_destination) {
    const DoubleReservoir source = make_source(5u, 2.0, 0.5);
    DoubleReservoir destination;
    const auto unsupported = restir::combine_reservoir(
        destination, source, 5u,
        restir::failed_shift<double>(
            restir::RestirShiftFailure::DestinationOutsideSupport),
        0.5);
    REQUIRE(unsupported.rejection ==
            restir::ReservoirRejectReason::ShiftDestinationOutsideSupport);
    REQUIRE(destination.M == 0u);

    auto invalid_jacobian = restir::identity_shift(2.0);
    invalid_jacobian.jacobian =
        std::numeric_limits<double>::quiet_NaN();
    const auto bad_jacobian = restir::combine_reservoir(
        destination, source, 5u, invalid_jacobian, 0.5);
    REQUIRE(bad_jacobian.rejection ==
            restir::ReservoirRejectReason::NonFiniteJacobian);
    REQUIRE(destination.M == 0u);

    const auto zero_mis = restir::combine_reservoir(
        destination, source, 5u, restir::identity_shift(2.0, 0.0), 0.5);
    REQUIRE(zero_mis.rejection ==
            restir::ReservoirRejectReason::ZeroMisWeight);
    REQUIRE(zero_mis.represented_candidates == source.M);
    REQUIRE(destination.M == source.M);
    REQUIRE(!restir::reservoir_has_sample(destination));
}

TEST_CASE(restir_validation_counters_classify_rejections) {
    restir::ReservoirValidationCounters counters;
    DoubleReservoir reservoir;
    counters.record(restir::stream_candidate(
        reservoir, restir::make_candidate(1u, 1.0, 1.0), 0.25));
    counters.record(restir::stream_candidate(
        reservoir,
        restir::make_candidate(2u, 0.0, 1.0, false, true), 0.25));
    REQUIRE(counters.accepted_operations == 1u);
    REQUIRE(counters.rejected_operations == 1u);
    REQUIRE(counters.represented_candidates == 2u);
    REQUIRE(counters.count(restir::ReservoirRejectReason::ZeroTarget) == 1u);
    REQUIRE(std::string(restir::reservoir_reject_reason_name(
                restir::ReservoirRejectReason::ZeroTarget)) ==
            "zero_target");
}

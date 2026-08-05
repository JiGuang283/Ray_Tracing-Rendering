#include "restir_reference.h"

#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct Options {
    std::uint64_t trials = 250000u;
    std::uint32_t seed = 1337u;
};

Options parse_options(int argc, char **argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto value = [&]() -> std::string {
            if (++index >= argc) {
                throw std::runtime_error(argument + " requires a value");
            }
            return argv[index];
        };
        if (argument == "--trials") {
            options.trials = std::stoull(value());
        } else if (argument == "--seed") {
            options.seed = static_cast<std::uint32_t>(std::stoul(value()));
        } else if (argument == "--help" || argument == "-h") {
            std::cout << "Usage: restir_reservoir_check [--trials N] "
                         "[--seed N]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + argument);
        }
    }
    if (options.trials == 0u) {
        throw std::runtime_error("--trials must be positive");
    }
    return options;
}

} // namespace

int main(int argc, char **argv) {
    try {
        const Options options = parse_options(argc, argv);
        const restir::RestirReferenceReport report =
            restir::run_restir_reference_validation(options.trials,
                                                     options.seed);
        std::cout << std::setprecision(9)
                  << "RESTIR_RESERVOIR_CHECK"
                  << " trials=" << report.trials
                  << " heavy_expected="
                  << report.heavy_expected_probability
                  << " heavy_observed="
                  << report.heavy_observed_probability
                  << " heavy_tolerance=" << report.heavy_tolerance
                  << " uniform_max_error=" << report.uniform_max_error
                  << " constant_expected="
                  << report.constant_expected_estimate
                  << " constant_observed="
                  << report.constant_mean_estimate
                  << " order_max_difference="
                  << report.forward_reverse_max_difference
                  << " result=" << (report.passed() ? "pass" : "fail")
                  << '\n';
        return report.passed() ? 0 : 1;
    } catch (const std::exception &error) {
        std::cerr << "RESTIR_RESERVOIR_ERROR message=" << error.what()
                  << '\n';
        return 1;
    }
}

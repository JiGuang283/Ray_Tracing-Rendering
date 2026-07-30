#ifndef TEST_HARNESS_H
#define TEST_HARNESS_H

#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

class TestRegistry {
  public:
    using Test = std::pair<std::string, std::function<void()>>;

    static TestRegistry &instance() {
        static TestRegistry registry;
        return registry;
    }

    bool add(std::string name, std::function<void()> test) {
        tests.emplace_back(std::move(name), std::move(test));
        return true;
    }

    int run() const {
        int failures = 0;
        for (const auto &test : tests) {
            try {
                test.second();
                std::cout << "[PASS] " << test.first << '\n';
            } catch (const std::exception &error) {
                ++failures;
                std::cerr << "[FAIL] " << test.first << ": " << error.what()
                          << '\n';
            }
        }
        std::cout << tests.size() - failures << "/" << tests.size()
                  << " tests passed\n";
        return failures == 0 ? 0 : 1;
    }

  private:
    std::vector<Test> tests;
};

#define TEST_CASE(name)                                                        \
    static void name();                                                        \
    static const bool name##_registered =                                      \
        TestRegistry::instance().add(#name, name);                             \
    static void name()

#define REQUIRE(condition)                                                     \
    do {                                                                       \
        if (!(condition)) {                                                    \
            throw std::runtime_error("requirement failed: " #condition);       \
        }                                                                      \
    } while (false)

#define REQUIRE_NEAR(actual, expected, tolerance)                              \
    do {                                                                       \
        const double actual_value = static_cast<double>(actual);               \
        const double expected_value = static_cast<double>(expected);           \
        if (std::abs(actual_value - expected_value) > (tolerance)) {            \
            throw std::runtime_error(                                          \
                "values differ: " + std::to_string(actual_value) + " vs " +   \
                std::to_string(expected_value));                               \
        }                                                                      \
    } while (false)

#endif

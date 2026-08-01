#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include "rng.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <thread>

using std::make_shared;
using std::make_unique;
using std::shared_ptr;
using std::sqrt;
using std::unique_ptr;

constexpr double infinity = std::numeric_limits<double>::infinity();
constexpr double pi = 3.1415926535897932385;

inline std::atomic<uint32_t> &random_seed_base() {
    static std::atomic<uint32_t> seed{0};
    return seed;
}

inline std::atomic<uint32_t> &random_seed_generation() {
    static std::atomic<uint32_t> generation{1};
    return generation;
}

inline void set_random_seed(uint32_t seed) {
    random_seed_base().store(seed == 0 ? 1 : seed, std::memory_order_relaxed);
    random_seed_generation().fetch_add(1, std::memory_order_relaxed);
}

inline uint32_t make_thread_seed() {
    uint32_t seed = random_seed_base().load(std::memory_order_relaxed);
    uint32_t thread_hash = static_cast<uint32_t>(
        std::hash<std::thread::id>{}(std::this_thread::get_id()));
    seed = seed == 0 ? thread_hash : (seed ^ thread_hash);
    return seed == 0 ? 1 : seed;
}

inline constexpr double degrees_to_radians(double degrees) {
    return degrees * pi / 180.0;
}

inline double random_double() {

    static thread_local uint32_t local_generation = 0;
    static thread_local uint32_t seed = make_thread_seed();

    uint32_t current_generation =
        random_seed_generation().load(std::memory_order_relaxed);
    if (local_generation != current_generation) {
        seed = make_thread_seed();
        local_generation = current_generation;
    }

    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;

    return seed * 2.3283064365386963e-10;
}

inline double random_double(double min, double max) noexcept {
    return min + (max - min) * random_double();
}

inline double clamp(double x, double min, double max) noexcept {
    if (x < min)
        return min;
    if (x > max)
        return max;
    return x;
}

inline int random_int(int min, int max) {
    return static_cast<int>(random_double(min, max + 1));
}

#endif

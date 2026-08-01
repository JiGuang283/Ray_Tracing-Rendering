#ifndef RNG_H
#define RNG_H

#include "host_device.h"

#include <cstdint>

struct RNG {
    std::uint32_t state;

    RT_HOST_DEVICE explicit RNG(std::uint32_t seed = 1)
        : state(seed == 0 ? 1 : seed) {
    }

    RT_HOST_DEVICE std::uint32_t next_u32() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    }

    RT_HOST_DEVICE double next() {
        return static_cast<double>(next_u32()) *
               2.3283064365386963e-10;
    }

    RT_HOST_DEVICE double next_double(double min, double max) {
        return min + (max - min) * next();
    }

    RT_HOST_DEVICE int next_int(int min, int max) {
        return static_cast<int>(next_double(min, max + 1));
    }
};

RT_HOST_DEVICE inline std::uint32_t mix_seed(std::uint32_t seed,
                                             std::uint32_t sequence) {
    std::uint32_t x = seed == 0 ? 1 : seed;
    x ^= sequence + 0x9e3779b9u + (x << 6) + (x >> 2);
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x == 0 ? 1 : x;
}

#endif

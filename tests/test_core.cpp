#include "rtweekend.h"
#include "test_harness.h"

TEST_CASE(rng_is_repeatable) {
    RNG first(123);
    RNG second(123);
    for (int i = 0; i < 32; ++i) {
        REQUIRE(first.next_u32() == second.next_u32());
    }
}

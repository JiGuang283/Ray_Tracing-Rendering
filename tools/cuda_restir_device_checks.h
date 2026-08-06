#ifndef CUDA_RESTIR_DEVICE_CHECKS_H
#define CUDA_RESTIR_DEVICE_CHECKS_H

#include <cstdint>

struct CudaReservoirDeviceCheckResult {
    std::uint32_t trials = 0;
    std::uint32_t failures = 0;
    double heavy_observed = 0.0;
    double tolerance = 0.0;
    bool algebra_pass = false;

    bool passed() const noexcept {
        return algebra_pass && failures == 0u &&
               heavy_observed >= 0.75 - tolerance &&
               heavy_observed <= 0.75 + tolerance;
    }
};

CudaReservoirDeviceCheckResult run_cuda_reservoir_device_check();

#endif

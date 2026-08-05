#include "gris_core.h"
#include "reservoir_core.h"
#include "rng.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using FloatReservoir = restir::Reservoir<std::uint32_t, float>;

struct DeviceAlgebraResult {
    FloatReservoir source;
    FloatReservoir combined;
    std::uint32_t status = 0;
};

void check_cuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

__global__ void algebra_kernel(DeviceAlgebraResult *result) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    DeviceAlgebraResult local;
    const auto first = restir::stream_candidate(
        local.source, restir::make_candidate(1u, 2.0f, 0.5f), 0.8f);
    const auto second = restir::stream_candidate(
        local.source, restir::make_candidate(2u, 6.0f, 0.5f), 0.1f);
    const auto source_final = restir::finalize_reservoir(local.source);

    auto shift = restir::identity_shift(3.0f, 0.25f);
    shift.jacobian = 2.0f;
    const auto combine = restir::combine_reservoir(
        local.combined, local.source, 9u, shift, 0.4f);
    const auto combined_final =
        restir::finalize_reservoir(local.combined);
    local.status =
        first.accepted() && second.accepted() && source_final.accepted() &&
                combine.accepted() && combined_final.accepted()
            ? 1u
            : 0u;
    *result = local;
}

DeviceAlgebraResult run_host_algebra() {
    DeviceAlgebraResult result;
    const auto first = restir::stream_candidate(
        result.source, restir::make_candidate(1u, 2.0f, 0.5f), 0.8f);
    const auto second = restir::stream_candidate(
        result.source, restir::make_candidate(2u, 6.0f, 0.5f), 0.1f);
    const auto source_final = restir::finalize_reservoir(result.source);
    auto shift = restir::identity_shift(3.0f, 0.25f);
    shift.jacobian = 2.0f;
    const auto combine = restir::combine_reservoir(
        result.combined, result.source, 9u, shift, 0.4f);
    const auto combined_final = restir::finalize_reservoir(result.combined);
    result.status =
        first.accepted() && second.accepted() && source_final.accepted() &&
                combine.accepted() && combined_final.accepted()
            ? 1u
            : 0u;
    return result;
}

__global__ void selection_kernel(std::uint32_t trials, std::uint32_t seed,
                                 std::uint32_t *heavy_count,
                                 std::uint32_t *failure_count) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= trials) {
        return;
    }
    RNG rng(mix_seed(seed, index));
    FloatReservoir reservoir;
    const auto first = restir::stream_candidate(
        reservoir, restir::make_candidate(0u, 1.0f, 1.0f),
        static_cast<float>(rng.next()));
    const auto second = restir::stream_candidate(
        reservoir, restir::make_candidate(1u, 3.0f, 1.0f),
        static_cast<float>(rng.next()));
    const auto final = restir::finalize_reservoir(reservoir);
    if (!first.accepted() || !second.accepted() || !final.accepted()) {
        atomicAdd(failure_count, 1u);
        return;
    }
    if (reservoir.sample == 1u) {
        atomicAdd(heavy_count, 1u);
    }
}

bool near(float left, float right, float tolerance = 1e-6f) {
    return std::abs(left - right) <= tolerance;
}

} // namespace

int main() {
    try {
        int device_count = 0;
        const cudaError_t device_status = cudaGetDeviceCount(&device_count);
        if (device_status != cudaSuccess || device_count == 0) {
            const char *reason = device_status == cudaSuccess
                                     ? "no CUDA device found"
                                     : cudaGetErrorString(device_status);
            std::cout << "CUDA_RESTIR_SKIP reason=" << reason << '\n';
            cudaGetLastError();
            return 77;
        }

        DeviceAlgebraResult *device_result = nullptr;
        check_cuda(cudaMalloc(&device_result, sizeof(DeviceAlgebraResult)),
                   "cudaMalloc algebra result");
        algebra_kernel<<<1, 1>>>(device_result);
        check_cuda(cudaGetLastError(), "launch algebra kernel");
        DeviceAlgebraResult algebra;
        check_cuda(cudaMemcpy(&algebra, device_result,
                              sizeof(DeviceAlgebraResult),
                              cudaMemcpyDeviceToHost),
                   "copy algebra result");
        check_cuda(cudaFree(device_result), "free algebra result");

        constexpr std::uint32_t trials = 262144u;
        std::uint32_t *device_counts = nullptr;
        check_cuda(cudaMalloc(&device_counts, 2u * sizeof(std::uint32_t)),
                   "cudaMalloc selection counts");
        check_cuda(cudaMemset(device_counts, 0, 2u * sizeof(std::uint32_t)),
                   "clear selection counts");
        constexpr std::uint32_t block_size = 256u;
        const std::uint32_t block_count =
            (trials + block_size - 1u) / block_size;
        selection_kernel<<<block_count, block_size>>>(
            trials, 123u, device_counts, device_counts + 1u);
        check_cuda(cudaGetLastError(), "launch selection kernel");
        std::uint32_t counts[2]{};
        check_cuda(cudaMemcpy(counts, device_counts, sizeof(counts),
                              cudaMemcpyDeviceToHost),
                   "copy selection counts");
        check_cuda(cudaFree(device_counts), "free selection counts");

        const double observed =
            static_cast<double>(counts[0]) / static_cast<double>(trials);
        const double tolerance =
            6.0 * std::sqrt(0.75 * 0.25 / static_cast<double>(trials)) +
            0.001;
        const DeviceAlgebraResult host_algebra = run_host_algebra();
        const bool algebra_pass =
            algebra.status == 1u && host_algebra.status == 1u &&
            algebra.source.sample == host_algebra.source.sample &&
            algebra.source.M == host_algebra.source.M &&
            near(algebra.source.weight_sum,
                 host_algebra.source.weight_sum) &&
            near(algebra.source.selected_target,
                 host_algebra.source.selected_target) &&
            near(algebra.source.unbiased_contribution_weight,
                 host_algebra.source.unbiased_contribution_weight) &&
            algebra.combined.sample == host_algebra.combined.sample &&
            algebra.combined.M == host_algebra.combined.M &&
            near(algebra.combined.weight_sum,
                 host_algebra.combined.weight_sum) &&
            near(algebra.combined.selected_target,
                 host_algebra.combined.selected_target) &&
            near(algebra.combined.unbiased_contribution_weight,
                 host_algebra.combined.unbiased_contribution_weight);
        const bool statistical_pass =
            counts[1] == 0u && std::abs(observed - 0.75) <= tolerance;
        const bool passed = algebra_pass && statistical_pass;
        std::cout << "CUDA_RESTIR_CHECK"
                  << " mode=reservoir"
                  << " trials=" << trials
                  << " heavy_observed=" << observed
                  << " tolerance=" << tolerance
                  << " failures=" << counts[1]
                  << " algebra=" << (algebra_pass ? "pass" : "fail")
                  << " result=" << (passed ? "pass" : "fail") << '\n';
        return passed ? 0 : 1;
    } catch (const std::exception &error) {
        std::cerr << "CUDA_RESTIR_ERROR message=" << error.what() << '\n';
        return 1;
    }
}

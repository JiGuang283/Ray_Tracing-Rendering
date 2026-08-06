#include "restir_initial_di.h"

#include "cuda_error.h"
#include "restir_di_core.h"

namespace cuda_backend {
namespace {

__device__ std::uint32_t status_index(restir::RestirDIStatus status) {
    const std::uint32_t index = static_cast<std::uint32_t>(status);
    return index < 11u
               ? index
               : static_cast<std::uint32_t>(
                     restir::RestirDIStatus::NonFinite);
}

__device__ float luminance(Float3 value) {
    return 0.2126f * value.x + 0.7152f * value.y + 0.0722f * value.z;
}

__global__ void generate_initial_di_candidates_kernel(
    DeviceSceneView scene, const restir::RestirSurface *surfaces,
    std::uint32_t width, std::uint32_t height,
    std::uint32_t iteration, std::uint32_t seed,
    std::uint32_t candidate_count,
    restir::RestirDIReservoir *reservoirs,
    DeviceRestirCounters *counters) {
    const std::uint32_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t pixel_count = width * height;
    if (pixel >= pixel_count) {
        return;
    }
    restir::RestirDICandidateStats stats;
    const restir::RestirDIStatus status =
        restir::generate_initial_di_reservoir(
            scene.scene, surfaces[pixel], width, height, pixel, iteration,
            seed, candidate_count, reservoirs[pixel], stats);
    atomicAdd(&counters->di_generation_status[status_index(status)], 1ull);
    atomicAdd(&counters->initial_candidates,
              static_cast<unsigned long long>(stats.attempted));
    atomicAdd(&counters->represented_candidates,
              static_cast<unsigned long long>(stats.represented));
    atomicAdd(&counters->rejected_candidates,
              static_cast<unsigned long long>(stats.rejected));
}

__global__ void shade_initial_di_kernel(
    DeviceSceneView scene, const restir::RestirSurface *surfaces,
    const restir::RestirDIReservoir *reservoirs, std::uint32_t width,
    std::uint32_t height, std::uint32_t iteration, std::uint32_t seed,
    float sample_clamp, CudaFilmPixel *film,
    DeviceRestirCounters *counters) {
    const std::uint32_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t pixel_count = width * height;
    if (pixel >= pixel_count) {
        return;
    }
    Float3 radiance{};
    std::uint32_t visibility_rays = 0;
    const restir::RestirDIStatus status =
        restir::shade_initial_di_reservoir(
            scene.scene, surfaces[pixel], reservoirs[pixel], width, height,
            pixel, iteration, seed, radiance, visibility_rays);
    if (restir::reservoir_is_usable(reservoirs[pixel])) {
        atomicAdd(&counters->valid_reservoirs, 1ull);
        atomicAdd(&counters->reservoir_M_sum,
                  static_cast<unsigned long long>(reservoirs[pixel].M));
        atomicAdd(&counters->reservoir_age_sum,
                  static_cast<unsigned long long>(reservoirs[pixel].age));
        atomicAdd(&counters->reservoir_effective_M_sum,
                  static_cast<double>(reservoirs[pixel].effective_M));
    }
    atomicAdd(&counters->visibility_rays,
              static_cast<unsigned long long>(visibility_rays));

    atomicAdd(&counters->di_shading_status[status_index(status)], 1ull);
    constexpr std::uint32_t kFallbackMask =
        restir::RESTIR_SURFACE_DELTA_ONLY |
        restir::RESTIR_SURFACE_UNSUPPORTED_DOMAIN;
    if ((surfaces[pixel].flags & kFallbackMask) != 0u) {
        radiance = {};
    }
    const bool expected_empty =
        status == restir::RestirDIStatus::NoSurface ||
        status == restir::RestirDIStatus::UnsupportedSurface ||
        status == restir::RestirDIStatus::ReservoirEmpty;
    if (status != restir::RestirDIStatus::Success && !expected_empty) {
        radiance = {};
        atomicAdd(&counters->di_invalid_samples, 1ull);
    } else if (!packed_transport::math::finite(radiance)) {
        radiance = {};
        atomicAdd(&counters->di_invalid_samples, 1ull);
    } else if (sample_clamp > 0.0f) {
        const float value = luminance(radiance);
        if (value > sample_clamp) {
            radiance = packed_transport::math::multiply(
                radiance, sample_clamp / value);
            atomicAdd(&counters->di_clamped_samples, 1ull);
        }
    }
    CudaFilmPixel &output = film[pixel];
    output.radiance = packed_transport::math::add(output.radiance,
                                                  radiance);
    ++output.sample_count;
}

} // namespace

void launch_restir_initial_di_candidates(
    DeviceSceneView scene, const restir::RestirSurface *surfaces,
    std::uint32_t width, std::uint32_t height,
    std::uint32_t iteration, std::uint32_t seed,
    std::uint32_t candidate_count,
    restir::RestirDIReservoir *reservoirs,
    DeviceRestirCounters *counters, std::uint32_t block_size) {
    const std::uint32_t pixel_count = width * height;
    const std::uint32_t grid =
        (pixel_count + block_size - 1u) / block_size;
    generate_initial_di_candidates_kernel<<<grid, block_size>>>(
        scene, surfaces, width, height, iteration, seed, candidate_count,
        reservoirs, counters);
    RT_CUDA_CHECK(cudaGetLastError());
}

void launch_restir_initial_di_shading(
    DeviceSceneView scene, const restir::RestirSurface *surfaces,
    const restir::RestirDIReservoir *reservoirs, std::uint32_t width,
    std::uint32_t height, std::uint32_t iteration, std::uint32_t seed,
    float sample_clamp, CudaFilmPixel *film,
    DeviceRestirCounters *counters, std::uint32_t block_size) {
    const std::uint32_t pixel_count = width * height;
    const std::uint32_t grid =
        (pixel_count + block_size - 1u) / block_size;
    shade_initial_di_kernel<<<grid, block_size>>>(
        scene, surfaces, reservoirs, width, height, iteration, seed,
        sample_clamp, film, counters);
    RT_CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda_backend

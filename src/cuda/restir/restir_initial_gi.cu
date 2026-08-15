#include "restir_initial_gi.h"

#include "cuda_error.h"
#include "restir_gi_core.h"

namespace cuda_backend {
namespace {

__device__ std::uint32_t gi_status_index(restir::RestirGIStatus status) {
    const std::uint32_t index = static_cast<std::uint32_t>(status);
    return index < 16u
               ? index
               : static_cast<std::uint32_t>(restir::RestirGIStatus::NonFinite);
}

__global__ void generate_initial_gi_candidates_kernel(
    DeviceSceneView scene, const restir::RestirSurface *surfaces,
    std::uint32_t width, std::uint32_t height,
    std::uint32_t iteration, std::uint32_t seed,
    std::uint32_t candidate_count, PackedTransportSettings transport,
    restir::RestirGIReservoir *reservoirs,
    CudaFilmPixel *fallback_film,
    DeviceRestirCounters *counters, std::uint32_t *status_output) {
    const std::uint32_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t pixel_count = width * height;
    if (pixel >= pixel_count) {
        return;
    }
    restir::RestirGICandidateStats stats;
    Float3 fallback_radiance{};
    const restir::RestirGIStatus status =
        restir::generate_initial_gi_reservoir(
            scene.scene, surfaces[pixel], width, height, pixel, iteration,
            seed, candidate_count, transport, reservoirs[pixel], stats,
            &fallback_radiance);
    if (fallback_film != nullptr) {
        fallback_film[pixel].radiance = packed_transport::math::add(
            fallback_film[pixel].radiance, fallback_radiance);
    }
    const std::uint32_t status_value = gi_status_index(status);
    if (status_output != nullptr) {
        status_output[pixel] = status_value;
    }
    atomicAdd(&counters->gi_generation_status[status_value], 1ull);
    atomicAdd(&counters->gi_initial_candidates,
              static_cast<unsigned long long>(stats.attempted));
    atomicAdd(&counters->gi_represented_candidates,
              static_cast<unsigned long long>(stats.represented));
    atomicAdd(&counters->gi_rejected_candidates,
              static_cast<unsigned long long>(stats.rejected));
    atomicAdd(&counters->gi_suffix_shadow_rays,
              static_cast<unsigned long long>(stats.suffix_shadow_rays));
    atomicAdd(&counters->gi_suffix_traversal_steps,
              static_cast<unsigned long long>(stats.suffix_traversal_steps));
    atomicAdd(&counters->gi_fallbacks,
              static_cast<unsigned long long>(stats.fallback_paths));
    atomicAdd(&counters->gi_replay_candidates,
              static_cast<unsigned long long>(stats.replay_candidates));
}

__global__ void shade_initial_gi_kernel(
    DeviceSceneView scene, const restir::RestirSurface *surfaces,
    const restir::RestirGIReservoir *reservoirs,
    std::uint32_t width, std::uint32_t height,
    std::uint32_t iteration, std::uint32_t seed,
    PackedTransportSettings transport,
    CudaFilmPixel *film, DeviceRestirCounters *counters,
    std::uint32_t *status_output, bool final_gather) {
    const std::uint32_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t pixel_count = width * height;
    if (pixel >= pixel_count) {
        return;
    }
    Float3 radiance{};
    restir::RestirGIShadingStats stats;
    restir::RestirGIShiftFailure failure =
        restir::RestirGIShiftFailure::None;
    const restir::RestirGIStatus status = restir::shade_gi_reservoir(
        scene.scene, surfaces[pixel], reservoirs[pixel], width, height,
        pixel, iteration, seed, radiance, transport, stats, failure,
        final_gather);
    const std::uint32_t status_value = gi_status_index(status);
    if (status_output != nullptr) {
        status_output[pixel] = status_value;
    }
    atomicAdd(&counters->gi_shading_status[status_value], 1ull);
    atomicAdd(&counters->gi_visibility_rays,
              static_cast<unsigned long long>(stats.visibility_rays));
    atomicAdd(&counters->gi_replay_evaluations,
              static_cast<unsigned long long>(stats.replay_evaluations));
    atomicAdd(&counters->gi_replay_shadow_rays,
              static_cast<unsigned long long>(stats.replay_shadow_rays));
    atomicAdd(&counters->gi_replay_traversal_steps,
              static_cast<unsigned long long>(
                  stats.replay_traversal_steps));
    if (restir::reservoir_is_usable(reservoirs[pixel])) {
        if (final_gather || reservoirs[pixel].sample.random_replay()) {
            atomicAdd(&counters->gi_replay_selections, 1ull);
        } else {
            atomicAdd(&counters->gi_reconnect_selections, 1ull);
        }
    }
    const std::uint32_t failure_index = static_cast<std::uint32_t>(failure);
    if (failure_index < 16u &&
        failure != restir::RestirGIShiftFailure::None) {
        atomicAdd(&counters->gi_shift_failures[failure_index], 1ull);
    }
    if (restir::reservoir_is_usable(reservoirs[pixel])) {
        atomicAdd(&counters->gi_valid_reservoirs, 1ull);
        atomicAdd(&counters->gi_reservoir_M_sum,
                  static_cast<unsigned long long>(reservoirs[pixel].M));
        atomicAdd(&counters->gi_reservoir_age_sum,
                  static_cast<unsigned long long>(reservoirs[pixel].age));
        atomicAdd(&counters->gi_reservoir_effective_M_sum,
                  static_cast<double>(reservoirs[pixel].effective_M));
    }
    const bool expected_empty =
        status == restir::RestirGIStatus::NoSurface ||
        status == restir::RestirGIStatus::UnsupportedPrimary ||
        status == restir::RestirGIStatus::ReservoirEmpty;
    if ((status != restir::RestirGIStatus::Success && !expected_empty) ||
        !packed_transport::math::finite(radiance)) {
        radiance = {};
        atomicAdd(&counters->gi_invalid_samples, 1ull);
    }
    CudaFilmPixel &output = film[pixel];
    output.radiance = packed_transport::math::add(output.radiance,
                                                  radiance);
    ++output.sample_count;
}

} // namespace

void launch_restir_initial_gi_candidates(
    DeviceSceneView scene, const restir::RestirSurface *surfaces,
    std::uint32_t width, std::uint32_t height,
    std::uint32_t iteration, std::uint32_t seed,
    std::uint32_t candidate_count,
    const PackedTransportSettings &transport,
    restir::RestirGIReservoir *reservoirs,
    CudaFilmPixel *fallback_film,
    DeviceRestirCounters *counters, std::uint32_t block_size,
    std::uint32_t *status_output) {
    const std::uint32_t pixel_count = width * height;
    const std::uint32_t grid =
        (pixel_count + block_size - 1u) / block_size;
    generate_initial_gi_candidates_kernel<<<grid, block_size>>>(
        scene, surfaces, width, height, iteration, seed, candidate_count,
        transport, reservoirs, fallback_film, counters, status_output);
    RT_CUDA_CHECK(cudaGetLastError());
}

void launch_restir_initial_gi_shading(
    DeviceSceneView scene, const restir::RestirSurface *surfaces,
    const restir::RestirGIReservoir *reservoirs,
    std::uint32_t width, std::uint32_t height,
    std::uint32_t iteration, std::uint32_t seed,
    const PackedTransportSettings &transport,
    CudaFilmPixel *film, DeviceRestirCounters *counters,
    std::uint32_t block_size, std::uint32_t *status_output,
    bool final_gather) {
    const std::uint32_t pixel_count = width * height;
    const std::uint32_t grid =
        (pixel_count + block_size - 1u) / block_size;
    shade_initial_gi_kernel<<<grid, block_size>>>(
        scene, surfaces, reservoirs, width, height, iteration, seed,
        transport, film, counters, status_output, final_gather);
    RT_CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda_backend

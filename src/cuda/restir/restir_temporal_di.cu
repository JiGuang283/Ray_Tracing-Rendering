#include "restir_temporal_di.h"

#include "cuda_error.h"
#include "restir_temporal_core.h"

namespace cuda_backend {
namespace {

__device__ std::uint32_t temporal_status_index(
    restir::RestirDIStatus status) {
    const std::uint32_t index = static_cast<std::uint32_t>(status);
    return index < 11u
               ? index
               : static_cast<std::uint32_t>(
                     restir::RestirDIStatus::NonFinite);
}

__global__ void temporal_di_kernel(
    DeviceSceneView scene, restir::RestirSurface *current_surfaces,
    const restir::RestirDIReservoir *current_reservoirs,
    const restir::RestirSurface *previous_surfaces,
    const restir::RestirDIReservoir *previous_reservoirs,
    PackedCamera previous_camera, bool history_available,
    restir::RestirDIReservoir *output, std::uint32_t width,
    std::uint32_t height, std::uint32_t iteration, std::uint32_t seed,
    std::uint32_t max_history_length, std::uint32_t max_candidates,
    float normal_threshold, float depth_threshold, bool pairwise,
    DeviceRestirCounters *counters) {
    __shared__ unsigned long long status_partial[11];
    __shared__ unsigned long long rejection_partial[16];
    __shared__ unsigned long long candidates_partial;
    __shared__ unsigned long long accepted_partial;
    __shared__ unsigned long long pairwise_partial;

    const unsigned thread = threadIdx.x;
    if (thread < 11u) {
        status_partial[thread] = 0;
    }
    if (thread < 16u) {
        rejection_partial[thread] = 0;
    }
    if (thread == 0) {
        candidates_partial = 0;
        accepted_partial = 0;
        pairwise_partial = 0;
    }
    __syncthreads();

    const std::uint32_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t pixel_count = width * height;
    if (pixel < pixel_count) {
        restir::RestirDITemporalStats stats;
        const PackedCamera *history_camera =
            history_available ? &previous_camera : nullptr;
        const restir::RestirSurface *history_surfaces =
            history_available ? previous_surfaces : nullptr;
        const restir::RestirDIReservoir *history_reservoirs =
            history_available ? previous_reservoirs : nullptr;
        const restir::RestirDIStatus status =
            pairwise
                ? restir::temporal_resample_di_pairwise(
                      scene.scene, current_surfaces[pixel],
                      current_reservoirs[pixel], history_camera,
                      history_surfaces, history_reservoirs, width, height,
                      pixel, iteration, seed, max_history_length,
                      max_candidates, normal_threshold, depth_threshold,
                      output[pixel], stats)
                : restir::temporal_resample_di_basic(
                      scene.scene, current_surfaces[pixel],
                      current_reservoirs[pixel], history_camera,
                      history_surfaces, history_reservoirs, width, height,
                      pixel, iteration, seed, max_history_length,
                      max_candidates, normal_threshold, depth_threshold,
                      output[pixel], stats);
        atomicAdd(&status_partial[temporal_status_index(status)], 1ull);
        atomicAdd(&rejection_partial[
                      static_cast<std::uint32_t>(stats.rejection)],
                  1ull);
        atomicAdd(&candidates_partial,
                  static_cast<unsigned long long>(stats.candidates));
        atomicAdd(&accepted_partial,
                  static_cast<unsigned long long>(stats.accepted));
        atomicAdd(&pairwise_partial,
                  static_cast<unsigned long long>(stats.pairwise_fallbacks));
    }

    __syncthreads();
    if (thread == 0) {
        for (unsigned index = 0; index < 11u; ++index) {
            atomicAdd(&counters->di_temporal_status[index],
                      status_partial[index]);
        }
        for (unsigned index = 0; index < 16u; ++index) {
            atomicAdd(&counters->temporal_rejection[index],
                      rejection_partial[index]);
        }
        atomicAdd(&counters->temporal_candidates, candidates_partial);
        atomicAdd(&counters->temporal_accepted, accepted_partial);
        atomicAdd(&counters->temporal_pairwise_fallbacks,
                  pairwise_partial);
    }
}

} // namespace

void launch_restir_temporal_di(
    DeviceSceneView scene, restir::RestirSurface *current_surfaces,
    const restir::RestirDIReservoir *current_reservoirs,
    const restir::RestirSurface *previous_surfaces,
    const restir::RestirDIReservoir *previous_reservoirs,
    PackedCamera previous_camera, bool history_available,
    restir::RestirDIReservoir *output, std::uint32_t width,
    std::uint32_t height, std::uint32_t iteration, std::uint32_t seed,
    std::uint32_t max_history_length, std::uint32_t max_candidates,
    float normal_threshold, float depth_threshold, bool pairwise,
    DeviceRestirCounters *counters, std::uint32_t block_size) {
    const std::uint32_t pixel_count = width * height;
    const std::uint32_t grid =
        (pixel_count + block_size - 1u) / block_size;
    temporal_di_kernel<<<grid, block_size>>>(
        scene, current_surfaces, current_reservoirs, previous_surfaces,
        previous_reservoirs, previous_camera, history_available, output,
        width, height, iteration, seed, max_history_length, max_candidates,
        normal_threshold, depth_threshold, pairwise, counters);
    RT_CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda_backend

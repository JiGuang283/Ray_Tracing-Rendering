#include "restir_gbuffer.h"

#include "cuda_error.h"
#include "restir_gbuffer_core.h"

namespace cuda_backend {
namespace {

__global__ void build_restir_gbuffer_kernel(
    DeviceSceneView scene, std::uint32_t width, std::uint32_t height,
    std::uint32_t iteration, std::uint32_t seed,
    restir::RestirSurface *output, DeviceRestirCounters *counters) {
    __shared__ unsigned long long status_partial[7];

    const unsigned thread = threadIdx.x;
    if (thread < 7u) {
        status_partial[thread] = 0;
    }
    __syncthreads();

    const std::uint32_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t pixel_count = width * height;
    if (pixel < pixel_count) {
        restir::RestirSurface surface;
        const restir::RestirGBufferStatus status =
            restir::build_primary_surface_core(
                scene.scene, width, height, pixel, iteration, seed, surface);
        output[pixel] = surface;
        std::uint32_t status_index = static_cast<std::uint32_t>(status);
        if (status_index >= 7u) {
            status_index =
                static_cast<std::uint32_t>(
                    restir::RestirGBufferStatus::InvalidInput);
        }
        atomicAdd(&status_partial[status_index], 1ull);
    }

    __syncthreads();
    if (thread == 0) {
        for (unsigned index = 0; index < 7u; ++index) {
            atomicAdd(&counters->gbuffer_status[index],
                      status_partial[index]);
        }
    }
}

} // namespace

void launch_restir_gbuffer(DeviceSceneView scene, std::uint32_t width,
                           std::uint32_t height,
                           std::uint32_t iteration,
                           std::uint32_t seed,
                           restir::RestirSurface *output,
                           DeviceRestirCounters *counters,
                           std::uint32_t block_size) {
    const std::uint32_t pixel_count = width * height;
    const std::uint32_t grid =
        (pixel_count + block_size - 1u) / block_size;
    build_restir_gbuffer_kernel<<<grid, block_size>>>(
        scene, width, height, iteration, seed, output, counters);
    RT_CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda_backend

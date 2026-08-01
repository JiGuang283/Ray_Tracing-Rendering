#include "intersection_kernels.h"

#include "render_data/flat_intersector_core.h"

#include <cuda_runtime.h>

#include <stdexcept>

namespace cuda_backend {
namespace {

class CudaEvent {
public:
    CudaEvent() {
        RT_CUDA_CHECK(cudaEventCreate(&m_event));
    }

    ~CudaEvent() {
        if (m_event != nullptr) {
            cudaEventDestroy(m_event);
        }
    }

    CudaEvent(const CudaEvent &) = delete;
    CudaEvent &operator=(const CudaEvent &) = delete;

    operator cudaEvent_t() const noexcept {
        return m_event;
    }

private:
    cudaEvent_t m_event = nullptr;
};

__global__ void intersect_rays_kernel(
    DeviceSceneView scene, const PackedRay *rays, PackedHit *hits,
    PackedTraversalStatus *status, std::uint32_t *rng_states,
    std::uint32_t ray_count) {
    const std::uint32_t index =
        blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= ray_count) {
        return;
    }
    PackedHit hit{};
    RNG rng(rng_states == nullptr ? 1u : rng_states[index]);
    RNG *rng_pointer = rng_states == nullptr ? nullptr : &rng;
    const PackedTraversalStatus result =
        packed_intersector::intersect_compiled_scene_core(
            scene.scene, rays[index], hit, rng_pointer);
    hits[index] = hit;
    status[index] = result;
    if (rng_states != nullptr) {
        rng_states[index] = rng.state;
    }
}

} // namespace

CudaIntersectionStats intersect_rays_cuda(
    DeviceSceneView scene, const PackedRay *device_rays,
    PackedHit *device_hits, PackedTraversalStatus *device_status,
    std::uint32_t *device_rng_states, std::uint32_t ray_count,
    std::uint32_t block_size) {
    if (ray_count == 0) {
        return {};
    }
    if (device_rays == nullptr || device_hits == nullptr ||
        device_status == nullptr) {
        throw std::invalid_argument(
            "CUDA intersection buffers must not be null");
    }
    if (block_size == 0 || block_size > 1024) {
        throw std::invalid_argument("invalid CUDA intersection block size");
    }

    CudaEvent begin;
    CudaEvent end;
    RT_CUDA_CHECK(cudaEventRecord(begin));
    const std::uint32_t grid_size =
        (ray_count + block_size - 1) / block_size;
    intersect_rays_kernel<<<grid_size, block_size>>>(
        scene, device_rays, device_hits, device_status, device_rng_states,
        ray_count);
    RT_CUDA_CHECK(cudaGetLastError());
    RT_CUDA_CHECK(cudaEventRecord(end));
    RT_CUDA_CHECK(cudaEventSynchronize(end));

    CudaIntersectionStats result;
    result.rays = ray_count;
    RT_CUDA_CHECK(cudaEventElapsedTime(&result.milliseconds, begin, end));
    return result;
}

} // namespace cuda_backend

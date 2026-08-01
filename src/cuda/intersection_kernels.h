#ifndef CUDA_INTERSECTION_KERNELS_H
#define CUDA_INTERSECTION_KERNELS_H

#include "device_scene.h"

#include <cstdint>

namespace cuda_backend {

struct CudaIntersectionStats {
    std::uint32_t rays = 0;
    float milliseconds = 0.0f;
};

CudaIntersectionStats intersect_rays_cuda(
    DeviceSceneView scene, const PackedRay *device_rays,
    PackedHit *device_hits, PackedTraversalStatus *device_status,
    std::uint32_t *device_rng_states, std::uint32_t ray_count,
    std::uint32_t block_size = 128);

} // namespace cuda_backend

#endif

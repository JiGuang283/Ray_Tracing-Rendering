#ifndef CUDA_LIGHT_KERNELS_H
#define CUDA_LIGHT_KERNELS_H

#include "device_scene.h"

#include <cstdint>

namespace cuda_backend {

struct CudaLightStageStats {
    std::uint32_t items = 0;
    float milliseconds = 0.0f;
};

CudaLightStageStats evaluate_lights_cuda(
    DeviceSceneView scene, const std::uint32_t *device_light_ids,
    const Float3 *device_origins, const Float2 *device_random_values,
    PackedLightSample *device_samples,
    PackedLightStatus *device_sample_status, float *device_pdfs,
    PackedLightStatus *device_pdf_status,
    std::uint32_t query_count, std::uint32_t block_size = 128);

CudaLightStageStats sample_non_delta_lights_cuda(
    DeviceSceneView scene, const Float3 *device_origins,
    std::uint32_t *device_rng_states,
    SelectedPackedLightSample *device_samples,
    PackedLightStatus *device_status, std::uint32_t query_count,
    std::uint32_t block_size = 128);

} // namespace cuda_backend

#endif

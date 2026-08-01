#ifndef CUDA_SHADING_KERNELS_H
#define CUDA_SHADING_KERNELS_H

#include "device_scene.h"

#include <cstdint>

namespace cuda_backend {

struct CudaShadingStageStats {
    std::uint32_t items = 0;
    float milliseconds = 0.0f;
};

CudaShadingStageStats reconstruct_hits_cuda(
    DeviceSceneView scene, const PackedRay *device_rays,
    const PackedHit *device_hits,
    const PackedTraversalStatus *device_traversal_status,
    PackedSurfaceInteraction *device_surfaces,
    PackedShadingStatus *device_reconstruction_status,
    std::uint32_t interaction_count, std::uint32_t block_size = 128);

CudaShadingStageStats evaluate_materials_cuda(
    DeviceSceneView scene,
    const PackedSurfaceInteraction *device_surfaces,
    const PackedShadingStatus *device_input_status,
    PackedMaterialOutput *device_material_outputs,
    PackedShadingStatus *device_material_status,
    std::uint32_t *device_texture_stack_usage,
    std::uint32_t interaction_count, std::uint32_t block_size = 128);

CudaShadingStageStats evaluate_bsdfs_cuda(
    const PackedMaterialOutput *device_material_outputs,
    const PackedShadingStatus *device_input_status,
    const Float3 *device_outgoing_directions,
    std::uint32_t *device_rng_states,
    PackedBSDFSample *device_samples,
    PackedBSDFStatus *device_sample_status,
    Float3 *device_evaluated_values, float *device_evaluated_pdfs,
    std::uint32_t interaction_count, std::uint32_t block_size = 128);

} // namespace cuda_backend

#endif

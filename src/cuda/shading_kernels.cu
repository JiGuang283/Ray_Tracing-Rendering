#include "shading_kernels.h"

#include "render_data/packed_bsdf_core.h"
#include "render_data/packed_material_core.h"
#include "render_data/surface_reconstruction_core.h"

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

__global__ void reconstruct_hits_kernel(
    DeviceSceneView scene, const PackedRay *rays, const PackedHit *hits,
    const PackedTraversalStatus *traversal_status,
    PackedSurfaceInteraction *surfaces,
    PackedShadingStatus *reconstruction_status,
    std::uint32_t interaction_count) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= interaction_count) {
        return;
    }

    PackedSurfaceInteraction surface{};
    PackedShadingStatus status = PackedShadingStatus::Miss;
    if (traversal_status[index] == PackedTraversalStatus::Hit) {
        status = packed_reconstruction::reconstruct_compiled_hit_core(
            scene.scene, rays[index], hits[index], surface);
    } else if (traversal_status[index] != PackedTraversalStatus::Miss) {
        status = PackedShadingStatus::InvalidInput;
    }
    surfaces[index] = surface;
    reconstruction_status[index] = status;
}

__global__ void evaluate_materials_kernel(
    DeviceSceneView scene, const PackedSurfaceInteraction *surfaces,
    const PackedShadingStatus *reconstruction_status,
    PackedMaterialOutput *outputs, PackedShadingStatus *material_status,
    std::uint32_t *texture_stack_usage,
    std::uint32_t interaction_count) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= interaction_count) {
        return;
    }

    PackedMaterialOutput output{};
    std::uint32_t stack_usage = 0;
    PackedShadingStatus status = reconstruction_status[index];
    if (status == PackedShadingStatus::Success) {
        const PackedSurfaceInteraction &surface = surfaces[index];
        status = packed_material::evaluate_packed_material_core(
            scene.scene, surface.material_id, surface, output,
            &stack_usage);
    }
    outputs[index] = output;
    material_status[index] = status;
    texture_stack_usage[index] = stack_usage;
}

__global__ void evaluate_bsdfs_kernel(
    const PackedMaterialOutput *material_outputs,
    const PackedShadingStatus *input_status,
    const Float3 *outgoing_directions, std::uint32_t *rng_states,
    PackedBSDFSample *samples, PackedBSDFStatus *sample_status,
    Float3 *evaluated_values, float *evaluated_pdfs,
    std::uint32_t interaction_count) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= interaction_count) {
        return;
    }

    PackedBSDFSample sample{};
    Float3 value{};
    float pdf = 0.0f;
    PackedBSDFStatus status = PackedBSDFStatus::InvalidInput;
    RNG rng(rng_states[index]);
    if (input_status[index] == PackedShadingStatus::Success) {
        status = packed_bsdf::sample_packed_bsdf_core(
            material_outputs[index], outgoing_directions[index], rng,
            sample);
        if (status == PackedBSDFStatus::Success) {
            const PackedBSDFStatus eval_status =
                packed_bsdf::eval_packed_bsdf_core(
                    material_outputs[index], outgoing_directions[index],
                    sample.wi, value);
            const PackedBSDFStatus pdf_status =
                packed_bsdf::pdf_packed_bsdf_core(
                    material_outputs[index], outgoing_directions[index],
                    sample.wi, pdf);
            if (eval_status != PackedBSDFStatus::Success ||
                pdf_status != PackedBSDFStatus::Success) {
                status = eval_status != PackedBSDFStatus::Success
                             ? eval_status
                             : pdf_status;
            }
        }
    }
    rng_states[index] = rng.state;
    samples[index] = sample;
    sample_status[index] = status;
    evaluated_values[index] = value;
    evaluated_pdfs[index] = pdf;
}

float elapsed_milliseconds(cudaEvent_t begin, cudaEvent_t end) {
    float result = 0.0f;
    RT_CUDA_CHECK(cudaEventElapsedTime(&result, begin, end));
    return result;
}

} // namespace

CudaShadingStageStats reconstruct_hits_cuda(
    DeviceSceneView scene, const PackedRay *device_rays,
    const PackedHit *device_hits,
    const PackedTraversalStatus *device_traversal_status,
    PackedSurfaceInteraction *device_surfaces,
    PackedShadingStatus *device_reconstruction_status,
    std::uint32_t interaction_count, std::uint32_t block_size) {
    if (interaction_count == 0) {
        return {};
    }
    if (device_rays == nullptr || device_hits == nullptr ||
        device_traversal_status == nullptr || device_surfaces == nullptr ||
        device_reconstruction_status == nullptr) {
        throw std::invalid_argument(
            "CUDA reconstruction buffers must not be null");
    }
    if (block_size == 0 || block_size > 1024) {
        throw std::invalid_argument(
            "invalid CUDA reconstruction block size");
    }

    const std::uint32_t grid_size =
        (interaction_count + block_size - 1) / block_size;
    CudaEvent begin;
    CudaEvent end;

    RT_CUDA_CHECK(cudaEventRecord(begin));
    reconstruct_hits_kernel<<<grid_size, block_size>>>(
        scene, device_rays, device_hits, device_traversal_status,
        device_surfaces, device_reconstruction_status,
        interaction_count);
    RT_CUDA_CHECK(cudaGetLastError());
    RT_CUDA_CHECK(cudaEventRecord(end));
    RT_CUDA_CHECK(cudaEventSynchronize(end));

    return {interaction_count, elapsed_milliseconds(begin, end)};
}

CudaShadingStageStats evaluate_materials_cuda(
    DeviceSceneView scene,
    const PackedSurfaceInteraction *device_surfaces,
    const PackedShadingStatus *device_input_status,
    PackedMaterialOutput *device_material_outputs,
    PackedShadingStatus *device_material_status,
    std::uint32_t *device_texture_stack_usage,
    std::uint32_t interaction_count, std::uint32_t block_size) {
    if (interaction_count == 0) {
        return {};
    }
    if (device_surfaces == nullptr || device_input_status == nullptr ||
        device_material_outputs == nullptr ||
        device_material_status == nullptr ||
        device_texture_stack_usage == nullptr) {
        throw std::invalid_argument(
            "CUDA material buffers must not be null");
    }
    if (block_size == 0 || block_size > 1024) {
        throw std::invalid_argument("invalid CUDA material block size");
    }

    const std::uint32_t grid_size =
        (interaction_count + block_size - 1) / block_size;
    CudaEvent begin;
    CudaEvent end;
    RT_CUDA_CHECK(cudaEventRecord(begin));
    evaluate_materials_kernel<<<grid_size, block_size>>>(
        scene, device_surfaces, device_input_status,
        device_material_outputs, device_material_status,
        device_texture_stack_usage, interaction_count);
    RT_CUDA_CHECK(cudaGetLastError());
    RT_CUDA_CHECK(cudaEventRecord(end));
    RT_CUDA_CHECK(cudaEventSynchronize(end));

    return {interaction_count, elapsed_milliseconds(begin, end)};
}

CudaShadingStageStats evaluate_bsdfs_cuda(
    const PackedMaterialOutput *device_material_outputs,
    const PackedShadingStatus *device_input_status,
    const Float3 *device_outgoing_directions,
    std::uint32_t *device_rng_states,
    PackedBSDFSample *device_samples,
    PackedBSDFStatus *device_sample_status,
    Float3 *device_evaluated_values, float *device_evaluated_pdfs,
    std::uint32_t interaction_count, std::uint32_t block_size) {
    if (interaction_count == 0) {
        return {};
    }
    if (device_material_outputs == nullptr ||
        device_input_status == nullptr ||
        device_outgoing_directions == nullptr ||
        device_rng_states == nullptr || device_samples == nullptr ||
        device_sample_status == nullptr ||
        device_evaluated_values == nullptr ||
        device_evaluated_pdfs == nullptr) {
        throw std::invalid_argument("CUDA BSDF buffers must not be null");
    }
    if (block_size == 0 || block_size > 1024) {
        throw std::invalid_argument("invalid CUDA BSDF block size");
    }

    const std::uint32_t grid_size =
        (interaction_count + block_size - 1) / block_size;
    CudaEvent begin;
    CudaEvent end;
    RT_CUDA_CHECK(cudaEventRecord(begin));
    evaluate_bsdfs_kernel<<<grid_size, block_size>>>(
        device_material_outputs, device_input_status,
        device_outgoing_directions, device_rng_states, device_samples,
        device_sample_status, device_evaluated_values,
        device_evaluated_pdfs, interaction_count);
    RT_CUDA_CHECK(cudaGetLastError());
    RT_CUDA_CHECK(cudaEventRecord(end));
    RT_CUDA_CHECK(cudaEventSynchronize(end));
    return {interaction_count, elapsed_milliseconds(begin, end)};
}

} // namespace cuda_backend

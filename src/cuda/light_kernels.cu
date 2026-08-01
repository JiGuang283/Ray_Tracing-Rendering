#include "light_kernels.h"

#include "render_data/packed_light_core.h"

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

__global__ void evaluate_lights_kernel(
    DeviceSceneView scene, const std::uint32_t *light_ids,
    const Float3 *origins, const Float2 *random_values,
    PackedLightSample *samples, PackedLightStatus *sample_status,
    float *pdfs, PackedLightStatus *pdf_status,
    std::uint32_t query_count) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= query_count) {
        return;
    }
    PackedLightSample sample{};
    const PackedLightStatus status =
        packed_light::sample_packed_light_core(
            scene.scene, light_ids[index], origins[index],
            random_values[index], sample);
    float pdf = 0.0f;
    const Float3 direction = status == PackedLightStatus::Success
                                 ? sample.wi
                                 : Float3{0.381f, 0.889f, -0.254f};
    const PackedLightStatus evaluated_pdf =
        packed_light::packed_light_pdf_core(
            scene.scene, light_ids[index], origins[index], direction, pdf);
    samples[index] = sample;
    sample_status[index] = status;
    pdfs[index] = pdf;
    pdf_status[index] = evaluated_pdf;
}

__global__ void sample_non_delta_lights_kernel(
    DeviceSceneView scene, const Float3 *origins,
    std::uint32_t *rng_states, SelectedPackedLightSample *samples,
    PackedLightStatus *statuses, std::uint32_t query_count) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= query_count) {
        return;
    }
    RNG rng(rng_states[index]);
    SelectedPackedLightSample sample{};
    const PackedLightStatus status =
        packed_light::sample_non_delta_light_core(
            scene.scene, origins[index], rng, sample);
    rng_states[index] = rng.state;
    samples[index] = sample;
    statuses[index] = status;
}

float elapsed_milliseconds(cudaEvent_t begin, cudaEvent_t end) {
    float result = 0.0f;
    RT_CUDA_CHECK(cudaEventElapsedTime(&result, begin, end));
    return result;
}

} // namespace

CudaLightStageStats evaluate_lights_cuda(
    DeviceSceneView scene, const std::uint32_t *device_light_ids,
    const Float3 *device_origins, const Float2 *device_random_values,
    PackedLightSample *device_samples,
    PackedLightStatus *device_sample_status, float *device_pdfs,
    PackedLightStatus *device_pdf_status, std::uint32_t query_count,
    std::uint32_t block_size) {
    if (query_count == 0) {
        return {};
    }
    if (device_light_ids == nullptr || device_origins == nullptr ||
        device_random_values == nullptr || device_samples == nullptr ||
        device_sample_status == nullptr || device_pdfs == nullptr ||
        device_pdf_status == nullptr) {
        throw std::invalid_argument("CUDA light buffers must not be null");
    }
    if (block_size == 0 || block_size > 1024) {
        throw std::invalid_argument("invalid CUDA light block size");
    }
    const std::uint32_t grid_size =
        (query_count + block_size - 1) / block_size;
    CudaEvent begin;
    CudaEvent end;
    RT_CUDA_CHECK(cudaEventRecord(begin));
    evaluate_lights_kernel<<<grid_size, block_size>>>(
        scene, device_light_ids, device_origins, device_random_values,
        device_samples, device_sample_status, device_pdfs,
        device_pdf_status, query_count);
    RT_CUDA_CHECK(cudaGetLastError());
    RT_CUDA_CHECK(cudaEventRecord(end));
    RT_CUDA_CHECK(cudaEventSynchronize(end));
    return {query_count, elapsed_milliseconds(begin, end)};
}

CudaLightStageStats sample_non_delta_lights_cuda(
    DeviceSceneView scene, const Float3 *device_origins,
    std::uint32_t *device_rng_states,
    SelectedPackedLightSample *device_samples,
    PackedLightStatus *device_status, std::uint32_t query_count,
    std::uint32_t block_size) {
    if (query_count == 0) {
        return {};
    }
    if (device_origins == nullptr || device_rng_states == nullptr ||
        device_samples == nullptr || device_status == nullptr) {
        throw std::invalid_argument(
            "CUDA non-delta light buffers must not be null");
    }
    if (block_size == 0 || block_size > 1024) {
        throw std::invalid_argument(
            "invalid CUDA non-delta light block size");
    }
    const std::uint32_t grid_size =
        (query_count + block_size - 1) / block_size;
    CudaEvent begin;
    CudaEvent end;
    RT_CUDA_CHECK(cudaEventRecord(begin));
    sample_non_delta_lights_kernel<<<grid_size, block_size>>>(
        scene, device_origins, device_rng_states, device_samples,
        device_status, query_count);
    RT_CUDA_CHECK(cudaGetLastError());
    RT_CUDA_CHECK(cudaEventRecord(end));
    RT_CUDA_CHECK(cudaEventSynchronize(end));
    return {query_count, elapsed_milliseconds(begin, end)};
}

} // namespace cuda_backend

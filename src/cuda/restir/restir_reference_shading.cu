#include "restir_reference_shading.h"

#include "cuda_error.h"
#include "packed_transport_core.h"

namespace cuda_backend {
namespace {

RT_HOST_DEVICE RT_FORCE_INLINE float luminance(Float3 value) noexcept {
    return 0.2126f * value.x + 0.7152f * value.y + 0.0722f * value.z;
}

__global__ void reference_shading_kernel(
    DeviceSceneView scene, PackedTransportSettings transport,
    std::uint32_t width, std::uint32_t height,
    std::uint32_t iteration, std::uint32_t seed, float sample_clamp,
    CudaFilmPixel *film, DeviceRestirCounters *counters) {
    const std::uint32_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t pixel_count = width * height;
    if (pixel >= pixel_count) {
        return;
    }
    RNG rng(packed_transport::packed_camera_sample_seed(seed, pixel,
                                                        iteration));
    const PackedRay ray =
        packed_transport::generate_packed_camera_ray_core(
            scene.scene.camera, pixel % width, pixel / width, width, height,
            rng);
    const PackedTransportResult traced =
        packed_transport::trace_packed_path_core(scene.scene, ray, transport,
                                                 rng);
    std::uint32_t status_index =
        static_cast<std::uint32_t>(traced.status);
    if (status_index >= 8u) {
        status_index = static_cast<std::uint32_t>(
            PackedTransportStatus::InvalidInput);
    }
    atomicAdd(&counters->transport_status[status_index], 1ull);
    atomicAdd(&counters->traversal_steps,
              static_cast<unsigned long long>(traced.traversal_steps));
    atomicAdd(&counters->shadow_rays,
              static_cast<unsigned long long>(traced.shadow_rays));

    Float3 radiance = traced.radiance;
    if (traced.status != PackedTransportStatus::Success ||
        !packed_transport::math::finite(radiance)) {
        radiance = {};
        atomicAdd(&counters->invalid_samples, 1ull);
    } else if (sample_clamp > 0.0f) {
        const float value = luminance(radiance);
        if (value > sample_clamp) {
            radiance = packed_transport::math::multiply(
                radiance, sample_clamp / value);
            atomicAdd(&counters->clamped_samples, 1ull);
        }
    }
    CudaFilmPixel &pixel_output = film[pixel];
    pixel_output.radiance = packed_transport::math::add(
        pixel_output.radiance, radiance);
    ++pixel_output.sample_count;
}

} // namespace

void launch_restir_reference_shading(
    DeviceSceneView scene, PackedTransportSettings transport,
    std::uint32_t width, std::uint32_t height,
    std::uint32_t iteration, std::uint32_t seed, float sample_clamp,
    CudaFilmPixel *film, DeviceRestirCounters *counters,
    std::uint32_t block_size) {
    const std::uint32_t pixel_count = width * height;
    const std::uint32_t grid =
        (pixel_count + block_size - 1u) / block_size;
    reference_shading_kernel<<<grid, block_size>>>(
        scene, transport, width, height, iteration, seed, sample_clamp, film,
        counters);
    RT_CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda_backend

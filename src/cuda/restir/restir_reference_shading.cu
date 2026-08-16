#include "restir_reference_shading.h"

#include "cuda_error.h"
#include "packed_transport_core.h"

namespace cuda_backend {
namespace {

static_assert(sizeof(DeviceRestirCounters) == 1880);

RT_HOST_DEVICE RT_FORCE_INLINE float luminance(Float3 value) noexcept {
    return 0.2126f * value.x + 0.7152f * value.y + 0.0722f * value.z;
}

__global__ void reference_shading_kernel(
    DeviceSceneView scene, PackedTransportSettings transport,
    std::uint32_t width, std::uint32_t height,
    std::uint32_t iteration, std::uint32_t seed, float sample_clamp,
    CudaFilmPixel *film, DeviceRestirCounters *counters,
    CudaRestirStatsLevel stats_level) {
    __shared__ unsigned long long status_partial[8];
    __shared__ unsigned long long traversal_partial;
    __shared__ unsigned long long shadow_partial;
    __shared__ unsigned long long invalid_partial;
    __shared__ unsigned long long clamped_partial;

    const bool collect_any = restir_collects_any_stats(stats_level);
    const bool collect_full = restir_collects_full_stats(stats_level);
    const unsigned thread = threadIdx.x;
    if (thread < 8u) {
        status_partial[thread] = 0;
    }
    if (thread == 0) {
        traversal_partial = 0;
        shadow_partial = 0;
        invalid_partial = 0;
        clamped_partial = 0;
    }
    __syncthreads();

    const std::uint32_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t pixel_count = width * height;
    if (pixel < pixel_count) {
        RNG rng(packed_transport::packed_camera_sample_seed(seed, pixel,
                                                            iteration));
        const PackedRay ray =
            packed_transport::generate_packed_camera_ray_core(
                scene.scene.camera, pixel % width, pixel / width, width,
                height, rng);
        const PackedTransportResult traced =
            packed_transport::trace_packed_path_core(scene.scene, ray,
                                                     transport, rng);
        std::uint32_t status_index =
            static_cast<std::uint32_t>(traced.status);
        if (status_index >= 8u) {
            status_index = static_cast<std::uint32_t>(
                PackedTransportStatus::InvalidInput);
        }
        if (collect_full) {
            atomicAdd(&status_partial[status_index], 1ull);
        }
        if (collect_any) {
            atomicAdd(&traversal_partial,
                      static_cast<unsigned long long>(
                          traced.traversal_steps));
            atomicAdd(&shadow_partial,
                      static_cast<unsigned long long>(
                          traced.shadow_rays));
        }

        Float3 radiance = traced.radiance;
        if (traced.status != PackedTransportStatus::Success ||
            !packed_transport::math::finite(radiance)) {
            radiance = {};
            if (collect_any) {
                atomicAdd(&invalid_partial, 1ull);
            }
        } else if (sample_clamp > 0.0f) {
            const float value = luminance(radiance);
            if (value > sample_clamp) {
                radiance = packed_transport::math::multiply(
                    radiance, sample_clamp / value);
                if (collect_any) {
                    atomicAdd(&clamped_partial, 1ull);
                }
            }
        }
        CudaFilmPixel &pixel_output = film[pixel];
        pixel_output.radiance = packed_transport::math::add(
            pixel_output.radiance, radiance);
        ++pixel_output.sample_count;
    }

    __syncthreads();
    if (thread == 0) {
        if (collect_full) {
            for (unsigned index = 0; index < 8u; ++index) {
                atomicAdd(&counters->transport_status[index],
                          status_partial[index]);
            }
        }
        if (collect_any) {
            atomicAdd(&counters->traversal_steps, traversal_partial);
            atomicAdd(&counters->shadow_rays, shadow_partial);
            atomicAdd(&counters->invalid_samples, invalid_partial);
            atomicAdd(&counters->clamped_samples, clamped_partial);
        }
    }
}

__global__ void fallback_shading_kernel(
    DeviceSceneView scene, PackedTransportSettings transport,
    const restir::RestirSurface *surfaces, std::uint32_t width,
    std::uint32_t height, std::uint32_t iteration, std::uint32_t seed,
    CudaFilmPixel *film, DeviceRestirCounters *counters,
    CudaRestirStatsLevel stats_level) {
    __shared__ unsigned long long invalid_partial;
    __shared__ unsigned long long fallback_partial;

    const bool collect_any = restir_collects_any_stats(stats_level);
    const unsigned thread = threadIdx.x;
    if (thread == 0) {
        invalid_partial = 0;
        fallback_partial = 0;
    }
    __syncthreads();

    const std::uint32_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t pixel_count = width * height;
    if (pixel < pixel_count) {
        const restir::RestirSurface &surface = surfaces[pixel];
        constexpr std::uint32_t kFallbackMask =
            restir::RESTIR_SURFACE_DELTA_ONLY |
            restir::RESTIR_SURFACE_UNSUPPORTED_DOMAIN;
        if (surface.valid() && (surface.flags & kFallbackMask) != 0u) {
            RNG rng(packed_transport::packed_camera_sample_seed(
                seed, pixel, iteration));
            const PackedRay ray =
                packed_transport::generate_packed_camera_ray_core(
                    scene.scene.camera, pixel % width, pixel / width, width,
                    height, rng);
            const PackedTransportResult traced =
                packed_transport::trace_packed_path_core(
                    scene.scene, ray, transport, rng);
            Float3 radiance = traced.radiance;
            if (traced.status != PackedTransportStatus::Success ||
                !packed_transport::math::finite(radiance)) {
                radiance = {};
                if (collect_any) {
                    atomicAdd(&invalid_partial, 1ull);
                }
            }
            film[pixel].radiance = packed_transport::math::add(
                film[pixel].radiance, radiance);
            if (collect_any) {
                atomicAdd(&fallback_partial, 1ull);
            }
        }
    }

    __syncthreads();
    if (thread == 0 && collect_any) {
        atomicAdd(&counters->gi_invalid_samples, invalid_partial);
        atomicAdd(&counters->gi_fallbacks, fallback_partial);
    }
}

} // namespace

void launch_restir_reference_shading(
    DeviceSceneView scene, PackedTransportSettings transport,
    std::uint32_t width, std::uint32_t height,
    std::uint32_t iteration, std::uint32_t seed, float sample_clamp,
    CudaFilmPixel *film, DeviceRestirCounters *counters,
    std::uint32_t block_size, CudaRestirStatsLevel stats_level) {
    const std::uint32_t pixel_count = width * height;
    const std::uint32_t grid =
        (pixel_count + block_size - 1u) / block_size;
    reference_shading_kernel<<<grid, block_size>>>(
        scene, transport, width, height, iteration, seed, sample_clamp, film,
        counters, stats_level);
    RT_CUDA_CHECK(cudaGetLastError());
}

void launch_restir_fallback_shading(
    DeviceSceneView scene, PackedTransportSettings transport,
    const restir::RestirSurface *surfaces, std::uint32_t width,
    std::uint32_t height, std::uint32_t iteration, std::uint32_t seed,
    CudaFilmPixel *film, DeviceRestirCounters *counters,
    std::uint32_t block_size, CudaRestirStatsLevel stats_level) {
    const std::uint32_t pixel_count = width * height;
    const std::uint32_t grid =
        (pixel_count + block_size - 1u) / block_size;
    fallback_shading_kernel<<<grid, block_size>>>(
        scene, transport, surfaces, width, height, iteration, seed, film,
        counters, stats_level);
    RT_CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda_backend

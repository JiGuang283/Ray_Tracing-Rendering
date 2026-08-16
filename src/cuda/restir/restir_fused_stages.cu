#include "restir_fused_stages.h"

#include "cuda_error.h"
#include "packed_transport_core.h"
#include "restir_di_core.h"
#include "restir_gbuffer_core.h"

namespace cuda_backend {
namespace {

__device__ std::uint32_t di_status_index(restir::RestirDIStatus status) {
    const std::uint32_t index = static_cast<std::uint32_t>(status);
    return index < 11u
               ? index
               : static_cast<std::uint32_t>(
                     restir::RestirDIStatus::NonFinite);
}

__device__ float luminance(Float3 value) {
    return 0.2126f * value.x + 0.7152f * value.y + 0.0722f * value.z;
}

__global__ void fused_gbuffer_initial_di_kernel(
    DeviceSceneView scene, std::uint32_t width, std::uint32_t height,
    std::uint32_t iteration, std::uint32_t seed,
    std::uint32_t candidate_count, restir::RestirSurface *surfaces,
    restir::RestirDIReservoir *reservoirs,
    DeviceRestirCounters *counters,
    CudaRestirStatsLevel stats_level) {
    __shared__ unsigned long long gbuffer_status_partial[7];
    __shared__ unsigned long long di_status_partial[11];
    __shared__ unsigned long long initial_partial;
    __shared__ unsigned long long represented_partial;
    __shared__ unsigned long long rejected_partial;

    const bool collect_any = restir_collects_any_stats(stats_level);
    const bool collect_full = restir_collects_full_stats(stats_level);
    const unsigned thread = threadIdx.x;
    if (thread < 7u) {
        gbuffer_status_partial[thread] = 0;
    }
    if (thread < 11u) {
        di_status_partial[thread] = 0;
    }
    if (thread == 0) {
        initial_partial = 0;
        represented_partial = 0;
        rejected_partial = 0;
    }
    __syncthreads();

    const std::uint32_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t pixel_count = width * height;
    if (pixel < pixel_count) {
        restir::RestirSurface surface;
        const restir::RestirGBufferStatus gbuffer_status =
            restir::build_primary_surface_core(
                scene.scene, width, height, pixel, iteration, seed, surface);
        surfaces[pixel] = surface;
        std::uint32_t gbuffer_status_index =
            static_cast<std::uint32_t>(gbuffer_status);
        if (gbuffer_status_index >= 7u) {
            gbuffer_status_index = static_cast<std::uint32_t>(
                restir::RestirGBufferStatus::InvalidInput);
        }
        if (collect_full) {
            atomicAdd(&gbuffer_status_partial[gbuffer_status_index], 1ull);
        }

        restir::RestirDICandidateStats stats;
        const restir::RestirDIStatus di_status =
            restir::generate_initial_di_reservoir(
                scene.scene, surface, width, height, pixel, iteration,
                seed, candidate_count, reservoirs[pixel], stats);
        if (collect_full) {
            atomicAdd(&di_status_partial[di_status_index(di_status)], 1ull);
        }
        if (collect_any) {
            atomicAdd(&initial_partial,
                      static_cast<unsigned long long>(stats.attempted));
            atomicAdd(&represented_partial,
                      static_cast<unsigned long long>(stats.represented));
            atomicAdd(&rejected_partial,
                      static_cast<unsigned long long>(stats.rejected));
        }
    }

    __syncthreads();
    if (thread == 0) {
        if (collect_full) {
            for (unsigned index = 0; index < 7u; ++index) {
                atomicAdd(&counters->gbuffer_status[index],
                          gbuffer_status_partial[index]);
            }
            for (unsigned index = 0; index < 11u; ++index) {
                atomicAdd(&counters->di_generation_status[index],
                          di_status_partial[index]);
            }
        }
        if (collect_any) {
            atomicAdd(&counters->initial_candidates, initial_partial);
            atomicAdd(&counters->represented_candidates,
                      represented_partial);
            atomicAdd(&counters->rejected_candidates, rejected_partial);
        }
    }
}

__global__ void fused_initial_di_fallback_shading_kernel(
    DeviceSceneView scene, PackedTransportSettings transport,
    const restir::RestirSurface *surfaces,
    const restir::RestirDIReservoir *reservoirs, std::uint32_t width,
    std::uint32_t height, std::uint32_t iteration, std::uint32_t seed,
    float sample_clamp, CudaFilmPixel *direct_film,
    DeviceRestirCounters *counters,
    CudaRestirStatsLevel stats_level) {
    __shared__ unsigned long long status_partial[11];
    __shared__ unsigned long long valid_partial;
    __shared__ unsigned long long M_partial;
    __shared__ unsigned long long age_partial;
    __shared__ double effective_M_partial;
    __shared__ unsigned long long visibility_partial;
    __shared__ unsigned long long di_invalid_partial;
    __shared__ unsigned long long clamped_partial;
    __shared__ unsigned long long fallback_invalid_partial;
    __shared__ unsigned long long fallback_partial;

    const bool collect_any = restir_collects_any_stats(stats_level);
    const bool collect_full = restir_collects_full_stats(stats_level);
    const unsigned thread = threadIdx.x;
    if (thread < 11u) {
        status_partial[thread] = 0;
    }
    if (thread == 0) {
        valid_partial = 0;
        M_partial = 0;
        age_partial = 0;
        effective_M_partial = 0.0;
        visibility_partial = 0;
        di_invalid_partial = 0;
        clamped_partial = 0;
        fallback_invalid_partial = 0;
        fallback_partial = 0;
    }
    __syncthreads();

    const std::uint32_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t pixel_count = width * height;
    if (pixel < pixel_count) {
        const restir::RestirSurface &surface = surfaces[pixel];
        CudaFilmPixel &direct = direct_film[pixel];

        // Initial DI reservoir shading, operation-for-operation identical to
        // launch_restir_initial_di_shading.
        Float3 radiance{};
        std::uint32_t visibility_rays = 0;
        const restir::RestirDIStatus status =
            restir::shade_initial_di_reservoir(
                scene.scene, surface, reservoirs[pixel], width, height,
                pixel, iteration, seed, radiance, visibility_rays);
        if (collect_any && restir::reservoir_is_usable(reservoirs[pixel])) {
            atomicAdd(&valid_partial, 1ull);
            atomicAdd(&M_partial,
                      static_cast<unsigned long long>(reservoirs[pixel].M));
            atomicAdd(&age_partial,
                      static_cast<unsigned long long>(reservoirs[pixel].age));
            atomicAdd(&effective_M_partial,
                      static_cast<double>(reservoirs[pixel].effective_M));
        }
        if (collect_any) {
            atomicAdd(&visibility_partial,
                      static_cast<unsigned long long>(visibility_rays));
        }
        if (collect_full) {
            atomicAdd(&status_partial[di_status_index(status)], 1ull);
        }

        constexpr std::uint32_t kFallbackMask =
            restir::RESTIR_SURFACE_DELTA_ONLY |
            restir::RESTIR_SURFACE_UNSUPPORTED_DOMAIN;
        if ((surface.flags & kFallbackMask) != 0u) {
            radiance = {};
        }
        const bool expected_empty =
            status == restir::RestirDIStatus::NoSurface ||
            status == restir::RestirDIStatus::UnsupportedSurface ||
            status == restir::RestirDIStatus::ReservoirEmpty;
        if (status != restir::RestirDIStatus::Success && !expected_empty) {
            radiance = {};
            if (collect_any) {
                atomicAdd(&di_invalid_partial, 1ull);
            }
        } else if (!packed_transport::math::finite(radiance)) {
            radiance = {};
            if (collect_any) {
                atomicAdd(&di_invalid_partial, 1ull);
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
        direct.radiance =
            packed_transport::math::add(direct.radiance, radiance);
        ++direct.sample_count;

        // Fallback shading, operation-for-operation identical to
        // launch_restir_fallback_shading. It intentionally does not touch
        // sample_count: initial shading owns the direct-film sample count.
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
            Float3 fallback_radiance = traced.radiance;
            if (traced.status != PackedTransportStatus::Success ||
                !packed_transport::math::finite(fallback_radiance)) {
                fallback_radiance = {};
                if (collect_any) {
                    atomicAdd(&fallback_invalid_partial, 1ull);
                }
            }
            direct.radiance = packed_transport::math::add(
                direct.radiance, fallback_radiance);
            if (collect_any) {
                atomicAdd(&fallback_partial, 1ull);
            }
        }
    }

    __syncthreads();
    if (thread == 0) {
        if (collect_full) {
            for (unsigned index = 0; index < 11u; ++index) {
                atomicAdd(&counters->di_shading_status[index],
                          status_partial[index]);
            }
        }
        if (collect_any) {
            atomicAdd(&counters->valid_reservoirs, valid_partial);
            atomicAdd(&counters->reservoir_M_sum, M_partial);
            atomicAdd(&counters->reservoir_age_sum, age_partial);
            atomicAdd(&counters->reservoir_effective_M_sum,
                      effective_M_partial);
            atomicAdd(&counters->visibility_rays, visibility_partial);
            atomicAdd(&counters->di_invalid_samples, di_invalid_partial);
            atomicAdd(&counters->di_clamped_samples, clamped_partial);
            atomicAdd(&counters->gi_invalid_samples,
                      fallback_invalid_partial);
            atomicAdd(&counters->gi_fallbacks, fallback_partial);
        }
    }
}

__global__ void fused_fallback_reference_shading_kernel(
    DeviceSceneView scene, PackedTransportSettings transport,
    const restir::RestirSurface *surfaces, std::uint32_t width,
    std::uint32_t height, std::uint32_t iteration, std::uint32_t seed,
    float sample_clamp, CudaFilmPixel *direct_film,
    CudaFilmPixel *reference_film, DeviceRestirCounters *counters,
    CudaRestirStatsLevel stats_level) {
    __shared__ unsigned long long status_partial[8];
    __shared__ unsigned long long traversal_partial;
    __shared__ unsigned long long shadow_partial;
    __shared__ unsigned long long invalid_partial;
    __shared__ unsigned long long clamped_partial;
    __shared__ unsigned long long fallback_invalid_partial;
    __shared__ unsigned long long fallback_partial;

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
        fallback_invalid_partial = 0;
        fallback_partial = 0;
    }
    __syncthreads();

    const std::uint32_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t pixel_count = width * height;
    if (pixel < pixel_count) {
        // Reference path, operation-for-operation identical to
        // launch_restir_reference_shading.
        RNG rng(packed_transport::packed_camera_sample_seed(
            seed, pixel, iteration));
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
        CudaFilmPixel &reference = reference_film[pixel];
        reference.radiance =
            packed_transport::math::add(reference.radiance, radiance);
        ++reference.sample_count;

        // Fallback path, operation-for-operation identical to
        // launch_restir_fallback_shading.
        const restir::RestirSurface &surface = surfaces[pixel];
        constexpr std::uint32_t kFallbackMask =
            restir::RESTIR_SURFACE_DELTA_ONLY |
            restir::RESTIR_SURFACE_UNSUPPORTED_DOMAIN;
        if (surface.valid() && (surface.flags & kFallbackMask) != 0u) {
            RNG fallback_rng(packed_transport::packed_camera_sample_seed(
                seed, pixel, iteration));
            const PackedRay fallback_ray =
                packed_transport::generate_packed_camera_ray_core(
                    scene.scene.camera, pixel % width, pixel / width, width,
                    height, fallback_rng);
            const PackedTransportResult fallback_traced =
                packed_transport::trace_packed_path_core(
                    scene.scene, fallback_ray, transport, fallback_rng);
            Float3 fallback_radiance = fallback_traced.radiance;
            if (fallback_traced.status != PackedTransportStatus::Success ||
                !packed_transport::math::finite(fallback_radiance)) {
                fallback_radiance = {};
                if (collect_any) {
                    atomicAdd(&fallback_invalid_partial, 1ull);
                }
            }
            CudaFilmPixel &direct = direct_film[pixel];
            direct.radiance = packed_transport::math::add(
                direct.radiance, fallback_radiance);
            if (collect_any) {
                atomicAdd(&fallback_partial, 1ull);
            }
        }
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
            atomicAdd(&counters->gi_invalid_samples,
                      fallback_invalid_partial);
            atomicAdd(&counters->gi_fallbacks, fallback_partial);
        }
    }
}

} // namespace

void launch_restir_fused_gbuffer_initial_di(
    DeviceSceneView scene, std::uint32_t width, std::uint32_t height,
    std::uint32_t iteration, std::uint32_t seed,
    std::uint32_t candidate_count, restir::RestirSurface *surfaces,
    restir::RestirDIReservoir *reservoirs, DeviceRestirCounters *counters,
    std::uint32_t block_size, CudaRestirStatsLevel stats_level) {
    const std::uint32_t pixel_count = width * height;
    const std::uint32_t grid =
        (pixel_count + block_size - 1u) / block_size;
    fused_gbuffer_initial_di_kernel<<<grid, block_size>>>(
        scene, width, height, iteration, seed, candidate_count, surfaces,
        reservoirs, counters, stats_level);
    RT_CUDA_CHECK(cudaGetLastError());
}

void launch_restir_fused_initial_di_fallback_shading(
    DeviceSceneView scene, PackedTransportSettings transport,
    const restir::RestirSurface *surfaces,
    const restir::RestirDIReservoir *reservoirs, std::uint32_t width,
    std::uint32_t height, std::uint32_t iteration, std::uint32_t seed,
    float sample_clamp, CudaFilmPixel *direct_film,
    DeviceRestirCounters *counters, std::uint32_t block_size,
    CudaRestirStatsLevel stats_level) {
    const std::uint32_t pixel_count = width * height;
    const std::uint32_t grid =
        (pixel_count + block_size - 1u) / block_size;
    fused_initial_di_fallback_shading_kernel<<<grid, block_size>>>(
        scene, transport, surfaces, reservoirs, width, height, iteration,
        seed, sample_clamp, direct_film, counters, stats_level);
    RT_CUDA_CHECK(cudaGetLastError());
}

void launch_restir_fused_fallback_reference_shading(
    DeviceSceneView scene, PackedTransportSettings transport,
    const restir::RestirSurface *surfaces, std::uint32_t width,
    std::uint32_t height, std::uint32_t iteration, std::uint32_t seed,
    float sample_clamp, CudaFilmPixel *direct_film,
    CudaFilmPixel *reference_film, DeviceRestirCounters *counters,
    std::uint32_t block_size, CudaRestirStatsLevel stats_level) {
    const std::uint32_t pixel_count = width * height;
    const std::uint32_t grid =
        (pixel_count + block_size - 1u) / block_size;
    fused_fallback_reference_shading_kernel<<<grid, block_size>>>(
        scene, transport, surfaces, width, height, iteration, seed,
        sample_clamp, direct_film, reference_film, counters, stats_level);
    RT_CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda_backend

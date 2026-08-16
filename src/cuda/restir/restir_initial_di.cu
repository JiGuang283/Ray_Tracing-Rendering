#include "restir_initial_di.h"

#include "cuda_error.h"
#include "restir_di_core.h"

namespace cuda_backend {
namespace {

__device__ std::uint32_t status_index(restir::RestirDIStatus status) {
    const std::uint32_t index = static_cast<std::uint32_t>(status);
    return index < 11u
               ? index
               : static_cast<std::uint32_t>(
                     restir::RestirDIStatus::NonFinite);
}

__device__ float luminance(Float3 value) {
    return 0.2126f * value.x + 0.7152f * value.y + 0.0722f * value.z;
}

__global__ void generate_initial_di_candidates_kernel(
    DeviceSceneView scene, const restir::RestirSurface *surfaces,
    std::uint32_t width, std::uint32_t height,
    std::uint32_t iteration, std::uint32_t seed,
    std::uint32_t candidate_count,
    restir::RestirDIReservoir *reservoirs,
    DeviceRestirCounters *counters,
    CudaRestirStatsLevel stats_level) {
    __shared__ unsigned long long status_partial[11];
    __shared__ unsigned long long initial_partial;
    __shared__ unsigned long long represented_partial;
    __shared__ unsigned long long rejected_partial;

    const bool collect_any = restir_collects_any_stats(stats_level);
    const bool collect_full = restir_collects_full_stats(stats_level);
    const unsigned thread = threadIdx.x;
    if (thread < 11u) {
        status_partial[thread] = 0;
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
        restir::RestirDICandidateStats stats;
        const restir::RestirDIStatus status =
            restir::generate_initial_di_reservoir(
                scene.scene, surfaces[pixel], width, height, pixel, iteration,
                seed, candidate_count, reservoirs[pixel], stats);
        if (collect_full) {
            atomicAdd(&status_partial[status_index(status)], 1ull);
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
            for (unsigned index = 0; index < 11u; ++index) {
                atomicAdd(&counters->di_generation_status[index],
                          status_partial[index]);
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

__global__ void shade_initial_di_kernel(
    DeviceSceneView scene, const restir::RestirSurface *surfaces,
    const restir::RestirDIReservoir *reservoirs, std::uint32_t width,
    std::uint32_t height, std::uint32_t iteration, std::uint32_t seed,
    float sample_clamp, CudaFilmPixel *film,
    DeviceRestirCounters *counters,
    CudaRestirStatsLevel stats_level) {
    __shared__ unsigned long long status_partial[11];
    __shared__ unsigned long long valid_partial;
    __shared__ unsigned long long M_partial;
    __shared__ unsigned long long age_partial;
    __shared__ double effective_M_partial;
    __shared__ unsigned long long visibility_partial;
    __shared__ unsigned long long invalid_partial;
    __shared__ unsigned long long clamped_partial;

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
        invalid_partial = 0;
        clamped_partial = 0;
    }
    __syncthreads();

    const std::uint32_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t pixel_count = width * height;
    if (pixel < pixel_count) {
        Float3 radiance{};
        std::uint32_t visibility_rays = 0;
        const restir::RestirDIStatus status =
            restir::shade_initial_di_reservoir(
                scene.scene, surfaces[pixel], reservoirs[pixel], width,
                height, pixel, iteration, seed, radiance, visibility_rays);
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
            atomicAdd(&status_partial[status_index(status)], 1ull);
        }
        constexpr std::uint32_t kFallbackMask =
            restir::RESTIR_SURFACE_DELTA_ONLY |
            restir::RESTIR_SURFACE_UNSUPPORTED_DOMAIN;
        if ((surfaces[pixel].flags & kFallbackMask) != 0u) {
            radiance = {};
        }
        const bool expected_empty =
            status == restir::RestirDIStatus::NoSurface ||
            status == restir::RestirDIStatus::UnsupportedSurface ||
            status == restir::RestirDIStatus::ReservoirEmpty;
        if (status != restir::RestirDIStatus::Success && !expected_empty) {
            radiance = {};
            if (collect_any) {
                atomicAdd(&invalid_partial, 1ull);
            }
        } else if (!packed_transport::math::finite(radiance)) {
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
        CudaFilmPixel &output = film[pixel];
        output.radiance =
            packed_transport::math::add(output.radiance, radiance);
        ++output.sample_count;
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
            atomicAdd(&counters->di_invalid_samples, invalid_partial);
            atomicAdd(&counters->di_clamped_samples, clamped_partial);
        }
    }
}

} // namespace

void launch_restir_initial_di_candidates(
    DeviceSceneView scene, const restir::RestirSurface *surfaces,
    std::uint32_t width, std::uint32_t height,
    std::uint32_t iteration, std::uint32_t seed,
    std::uint32_t candidate_count,
    restir::RestirDIReservoir *reservoirs,
    DeviceRestirCounters *counters, std::uint32_t block_size,
    CudaRestirStatsLevel stats_level) {
    const std::uint32_t pixel_count = width * height;
    const std::uint32_t grid =
        (pixel_count + block_size - 1u) / block_size;
    generate_initial_di_candidates_kernel<<<grid, block_size>>>(
        scene, surfaces, width, height, iteration, seed, candidate_count,
        reservoirs, counters, stats_level);
    RT_CUDA_CHECK(cudaGetLastError());
}

void launch_restir_initial_di_shading(
    DeviceSceneView scene, const restir::RestirSurface *surfaces,
    const restir::RestirDIReservoir *reservoirs, std::uint32_t width,
    std::uint32_t height, std::uint32_t iteration, std::uint32_t seed,
    float sample_clamp, CudaFilmPixel *film,
    DeviceRestirCounters *counters, std::uint32_t block_size,
    CudaRestirStatsLevel stats_level) {
    const std::uint32_t pixel_count = width * height;
    const std::uint32_t grid =
        (pixel_count + block_size - 1u) / block_size;
    shade_initial_di_kernel<<<grid, block_size>>>(
        scene, surfaces, reservoirs, width, height, iteration, seed,
        sample_clamp, film, counters, stats_level);
    RT_CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda_backend

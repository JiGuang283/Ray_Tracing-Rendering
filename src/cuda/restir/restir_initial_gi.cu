#include "restir_initial_gi.h"

#include "cuda_error.h"
#include "restir_gi_core.h"

namespace cuda_backend {
namespace {

__device__ std::uint32_t gi_status_index(restir::RestirGIStatus status) {
    const std::uint32_t index = static_cast<std::uint32_t>(status);
    return index < 16u
               ? index
               : static_cast<std::uint32_t>(restir::RestirGIStatus::NonFinite);
}

__device__ float luminance(Float3 value) {
    return 0.2126f * value.x + 0.7152f * value.y + 0.0722f * value.z;
}

__global__ void generate_initial_gi_candidates_kernel(
    DeviceSceneView scene, const restir::RestirSurface *surfaces,
    std::uint32_t width, std::uint32_t height,
    std::uint32_t iteration, std::uint32_t seed,
    std::uint32_t candidate_count, PackedTransportSettings transport,
    restir::RestirGIReservoir *reservoirs,
    CudaFilmPixel *fallback_film,
    DeviceRestirCounters *counters, std::uint32_t *status_output,
    CudaRestirStatsLevel stats_level) {
    __shared__ unsigned long long status_partial[16];
    __shared__ unsigned long long initial_partial;
    __shared__ unsigned long long represented_partial;
    __shared__ unsigned long long rejected_partial;
    __shared__ unsigned long long suffix_shadow_partial;
    __shared__ unsigned long long suffix_traversal_partial;
    __shared__ unsigned long long fallback_partial;
    __shared__ unsigned long long replay_partial;

    const bool collect_any = restir_collects_any_stats(stats_level);
    const bool collect_full = restir_collects_full_stats(stats_level);
    const unsigned thread = threadIdx.x;
    if (thread < 16u) {
        status_partial[thread] = 0;
    }
    if (thread == 0) {
        initial_partial = 0;
        represented_partial = 0;
        rejected_partial = 0;
        suffix_shadow_partial = 0;
        suffix_traversal_partial = 0;
        fallback_partial = 0;
        replay_partial = 0;
    }
    __syncthreads();

    const std::uint32_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t pixel_count = width * height;
    if (pixel < pixel_count) {
        restir::RestirGICandidateStats stats;
        Float3 fallback_radiance{};
        const restir::RestirGIStatus status =
            restir::generate_initial_gi_reservoir(
                scene.scene, surfaces[pixel], width, height, pixel, iteration,
                seed, candidate_count, transport, reservoirs[pixel], stats,
                &fallback_radiance);
        if (fallback_film != nullptr) {
            fallback_film[pixel].radiance = packed_transport::math::add(
                fallback_film[pixel].radiance, fallback_radiance);
        }
        const std::uint32_t status_value = gi_status_index(status);
        if (status_output != nullptr) {
            status_output[pixel] = status_value;
        }
        if (collect_full) {
            atomicAdd(&status_partial[status_value], 1ull);
        }
        if (collect_any) {
            atomicAdd(&initial_partial,
                      static_cast<unsigned long long>(stats.attempted));
            atomicAdd(&represented_partial,
                      static_cast<unsigned long long>(stats.represented));
            atomicAdd(&rejected_partial,
                      static_cast<unsigned long long>(stats.rejected));
            atomicAdd(&suffix_shadow_partial,
                      static_cast<unsigned long long>(
                          stats.suffix_shadow_rays));
            atomicAdd(&suffix_traversal_partial,
                      static_cast<unsigned long long>(
                          stats.suffix_traversal_steps));
            atomicAdd(&fallback_partial,
                      static_cast<unsigned long long>(stats.fallback_paths));
            atomicAdd(&replay_partial,
                      static_cast<unsigned long long>(
                          stats.replay_candidates));
        }
    }

    __syncthreads();
    if (thread == 0) {
        if (collect_full) {
            for (unsigned index = 0; index < 16u; ++index) {
                atomicAdd(&counters->gi_generation_status[index],
                          status_partial[index]);
            }
        }
        if (collect_any) {
            atomicAdd(&counters->gi_initial_candidates, initial_partial);
            atomicAdd(&counters->gi_represented_candidates,
                      represented_partial);
            atomicAdd(&counters->gi_rejected_candidates,
                      rejected_partial);
            atomicAdd(&counters->gi_suffix_shadow_rays,
                      suffix_shadow_partial);
            atomicAdd(&counters->gi_suffix_traversal_steps,
                      suffix_traversal_partial);
            atomicAdd(&counters->gi_fallbacks, fallback_partial);
            atomicAdd(&counters->gi_replay_candidates, replay_partial);
        }
    }
}

__global__ void shade_initial_gi_kernel(
    DeviceSceneView scene, const restir::RestirSurface *surfaces,
    const restir::RestirGIReservoir *reservoirs,
    std::uint32_t width, std::uint32_t height,
    std::uint32_t iteration, std::uint32_t seed,
    PackedTransportSettings transport,
    CudaFilmPixel *film, DeviceRestirCounters *counters,
    float sample_clamp, std::uint32_t *status_output,
    bool final_gather, CudaRestirStatsLevel stats_level) {
    __shared__ unsigned long long status_partial[16];
    __shared__ unsigned long long shift_partial[16];
    __shared__ unsigned long long visibility_partial;
    __shared__ unsigned long long replay_eval_partial;
    __shared__ unsigned long long replay_shadow_partial;
    __shared__ unsigned long long replay_traversal_partial;
    __shared__ unsigned long long replay_selection_partial;
    __shared__ unsigned long long reconnect_selection_partial;
    __shared__ unsigned long long valid_partial;
    __shared__ unsigned long long M_partial;
    __shared__ unsigned long long age_partial;
    __shared__ double effective_M_partial;
    __shared__ unsigned long long invalid_partial;
    __shared__ unsigned long long clamped_partial;

    const bool collect_any = restir_collects_any_stats(stats_level);
    const bool collect_full = restir_collects_full_stats(stats_level);
    const unsigned thread = threadIdx.x;
    if (thread < 16u) {
        status_partial[thread] = 0;
        shift_partial[thread] = 0;
    }
    if (thread == 0) {
        visibility_partial = 0;
        replay_eval_partial = 0;
        replay_shadow_partial = 0;
        replay_traversal_partial = 0;
        replay_selection_partial = 0;
        reconnect_selection_partial = 0;
        valid_partial = 0;
        M_partial = 0;
        age_partial = 0;
        effective_M_partial = 0.0;
        invalid_partial = 0;
        clamped_partial = 0;
    }
    __syncthreads();

    const std::uint32_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t pixel_count = width * height;
    if (pixel < pixel_count) {
        Float3 radiance{};
        restir::RestirGIShadingStats stats;
        restir::RestirGIShiftFailure failure =
            restir::RestirGIShiftFailure::None;
        const restir::RestirGIStatus status = restir::shade_gi_reservoir(
            scene.scene, surfaces[pixel], reservoirs[pixel], width, height,
            pixel, iteration, seed, radiance, transport, stats, failure,
            final_gather);
        const std::uint32_t status_value = gi_status_index(status);
        if (status_output != nullptr) {
            status_output[pixel] = status_value;
        }
        if (collect_full) {
            atomicAdd(&status_partial[status_value], 1ull);
        }
        if (collect_any) {
            atomicAdd(&visibility_partial,
                      static_cast<unsigned long long>(
                          stats.visibility_rays));
            atomicAdd(&replay_eval_partial,
                      static_cast<unsigned long long>(
                          stats.replay_evaluations));
            atomicAdd(&replay_shadow_partial,
                      static_cast<unsigned long long>(
                          stats.replay_shadow_rays));
            atomicAdd(&replay_traversal_partial,
                      static_cast<unsigned long long>(
                          stats.replay_traversal_steps));
            if (restir::reservoir_is_usable(reservoirs[pixel])) {
                if (final_gather ||
                    reservoirs[pixel].sample.random_replay()) {
                    atomicAdd(&replay_selection_partial, 1ull);
                } else {
                    atomicAdd(&reconnect_selection_partial, 1ull);
                }
            }
            if (restir::reservoir_is_usable(reservoirs[pixel])) {
                atomicAdd(&valid_partial, 1ull);
                atomicAdd(&M_partial, static_cast<unsigned long long>(
                                         reservoirs[pixel].M));
                atomicAdd(&age_partial,
                          static_cast<unsigned long long>(
                              reservoirs[pixel].age));
                atomicAdd(&effective_M_partial,
                          static_cast<double>(
                              reservoirs[pixel].effective_M));
            }
        }
        const std::uint32_t failure_index =
            static_cast<std::uint32_t>(failure);
        if (collect_full && failure_index < 16u &&
            failure != restir::RestirGIShiftFailure::None) {
            atomicAdd(&shift_partial[failure_index], 1ull);
        }
        const bool expected_empty =
            status == restir::RestirGIStatus::NoSurface ||
            status == restir::RestirGIStatus::UnsupportedPrimary ||
            status == restir::RestirGIStatus::ReservoirEmpty;
        if ((status != restir::RestirGIStatus::Success && !expected_empty) ||
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
        CudaFilmPixel &output = film[pixel];
        output.radiance =
            packed_transport::math::add(output.radiance, radiance);
        ++output.sample_count;
    }

    __syncthreads();
    if (thread == 0) {
        if (collect_full) {
            for (unsigned index = 0; index < 16u; ++index) {
                atomicAdd(&counters->gi_shading_status[index],
                          status_partial[index]);
                atomicAdd(&counters->gi_shift_failures[index],
                          shift_partial[index]);
            }
        }
        if (collect_any) {
            atomicAdd(&counters->gi_visibility_rays, visibility_partial);
            atomicAdd(&counters->gi_replay_evaluations,
                      replay_eval_partial);
            atomicAdd(&counters->gi_replay_shadow_rays,
                      replay_shadow_partial);
            atomicAdd(&counters->gi_replay_traversal_steps,
                      replay_traversal_partial);
            atomicAdd(&counters->gi_replay_selections,
                      replay_selection_partial);
            atomicAdd(&counters->gi_reconnect_selections,
                      reconnect_selection_partial);
            atomicAdd(&counters->gi_valid_reservoirs, valid_partial);
            atomicAdd(&counters->gi_reservoir_M_sum, M_partial);
            atomicAdd(&counters->gi_reservoir_age_sum, age_partial);
            atomicAdd(&counters->gi_reservoir_effective_M_sum,
                      effective_M_partial);
            atomicAdd(&counters->gi_invalid_samples, invalid_partial);
            atomicAdd(&counters->gi_clamped_samples, clamped_partial);
        }
    }
}

} // namespace

void launch_restir_initial_gi_candidates(
    DeviceSceneView scene, const restir::RestirSurface *surfaces,
    std::uint32_t width, std::uint32_t height,
    std::uint32_t iteration, std::uint32_t seed,
    std::uint32_t candidate_count,
    const PackedTransportSettings &transport,
    restir::RestirGIReservoir *reservoirs,
    CudaFilmPixel *fallback_film,
    DeviceRestirCounters *counters, std::uint32_t block_size,
    std::uint32_t *status_output, CudaRestirStatsLevel stats_level) {
    const std::uint32_t pixel_count = width * height;
    const std::uint32_t grid =
        (pixel_count + block_size - 1u) / block_size;
    generate_initial_gi_candidates_kernel<<<grid, block_size>>>(
        scene, surfaces, width, height, iteration, seed, candidate_count,
        transport, reservoirs, fallback_film, counters, status_output,
        stats_level);
    RT_CUDA_CHECK(cudaGetLastError());
}

void launch_restir_initial_gi_shading(
    DeviceSceneView scene, const restir::RestirSurface *surfaces,
    const restir::RestirGIReservoir *reservoirs,
    std::uint32_t width, std::uint32_t height,
    std::uint32_t iteration, std::uint32_t seed,
    const PackedTransportSettings &transport,
    CudaFilmPixel *film, DeviceRestirCounters *counters,
    std::uint32_t block_size, float sample_clamp,
    std::uint32_t *status_output, bool final_gather,
    CudaRestirStatsLevel stats_level) {
    const std::uint32_t pixel_count = width * height;
    const std::uint32_t grid =
        (pixel_count + block_size - 1u) / block_size;
    shade_initial_gi_kernel<<<grid, block_size>>>(
        scene, surfaces, reservoirs, width, height, iteration, seed,
        transport, film, counters, sample_clamp, status_output,
        final_gather, stats_level);
    RT_CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda_backend

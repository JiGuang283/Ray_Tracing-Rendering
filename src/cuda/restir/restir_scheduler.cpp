#include "restir_scheduler.h"

#include "cuda_error.h"
#include "restir_gbuffer.h"
#include "restir_initial_di.h"
#include "restir_initial_gi.h"
#include "restir_reference_shading.h"
#include "restir_spatial_di.h"
#include "restir_spatial_gi.h"
#include "restir_temporal_di.h"
#include "restir_temporal_gi.h"
#include "restir_workspace_internal.h"

#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace cuda_backend {
namespace {

class SchedulerEvent {
  public:
    SchedulerEvent() {
        RT_CUDA_CHECK(cudaEventCreate(&m_event));
    }

    ~SchedulerEvent() {
        if (m_event != nullptr) {
            cudaEventDestroy(m_event);
        }
    }

    SchedulerEvent(const SchedulerEvent &) = delete;
    SchedulerEvent &operator=(const SchedulerEvent &) = delete;

    operator cudaEvent_t() const noexcept {
        return m_event;
    }

  private:
    cudaEvent_t m_event = nullptr;
};

void validate_settings(const CudaRestirSkeletonSettings &settings) {
    validate_render_frame_request(settings.frame);
    const IntegratorDescriptor &descriptor =
        integrator_descriptor(settings.frame.render.integrator);
    if (descriptor.execution_model !=
        IntegratorExecutionModel::RestirFrame) {
        throw std::invalid_argument(
            "ReSTIR scheduler requires a RestirFrame integrator");
    }
    if (settings.generate_reference &&
        (!valid_integrator_policy(settings.reference_transport.policy) ||
         settings.reference_transport.max_depth == 0u)) {
        throw std::invalid_argument(
            "invalid ReSTIR skeleton reference transport policy");
    }
    if (settings.block_size == 0u || settings.block_size > 1024u) {
        throw std::invalid_argument(
            "ReSTIR block size must be in the range 1..1024");
    }
    if (settings.frame.render.restir.initial_light_candidates == 0u) {
        throw std::invalid_argument(
            "initial ReSTIR DI requires nonzero light candidates");
    }
    if (settings.frame.render.restir.initial_bsdf_candidates != 0u) {
        throw std::invalid_argument(
            "BSDF-generated ReSTIR DI candidates are not implemented");
    }
    const bool gi_mode =
        settings.frame.render.integrator == IntegratorKind::ReSTIRGI;
    if (gi_mode &&
        settings.frame.render.restir.initial_gi_candidates == 0u) {
        throw std::invalid_argument(
            "initial ReSTIR GI requires nonzero path candidates");
    }
    if (gi_mode && settings.frame.render.max_depth < 2u) {
        throw std::invalid_argument(
            "diffuse ReSTIR GI requires a maximum depth of at least 2");
    }
    if (gi_mode &&
        settings.frame.render.restir.bias_correction !=
            RestirBiasCorrection::Basic) {
        throw std::invalid_argument(
            "diffuse ReSTIR GI currently requires basic correction");
    }
    if ((settings.frame.render.restir.spatial_reuse ||
         settings.frame.render.restir.temporal_reuse) &&
        settings.frame.render.restir.bias_correction ==
            RestirBiasCorrection::Unbiased) {
        throw std::invalid_argument(
            "unbiased ReSTIR DI spatial correction is not implemented");
    }
    if (!std::isfinite(settings.frame.render.sample_clamp) ||
        settings.frame.render.sample_clamp < 0.0) {
        throw std::invalid_argument("invalid ReSTIR sample clamp");
    }
}

std::uint32_t select_di_scratch(std::uint32_t source,
                                std::uint32_t protected_history) {
    for (std::uint32_t index = 0u;
         index < restir::kRestirDIReservoirBufferCount; ++index) {
        if (index != source && index != protected_history) {
            return index;
        }
    }
    throw std::logic_error("no transactional ReSTIR DI scratch buffer");
}

std::uint32_t select_gi_scratch(std::uint32_t source,
                                std::uint32_t protected_history) {
    for (std::uint32_t index = 0u;
         index < restir::kRestirGIReservoirBufferCount; ++index) {
        if (index != source && index != protected_history) {
            return index;
        }
    }
    throw std::logic_error("no transactional ReSTIR GI scratch buffer");
}

} // namespace

CudaRestirSchedulerOutput render_restir_skeleton_cuda(
    DeviceSceneView scene, const CudaRestirSkeletonSettings &settings,
    CudaRestirWorkspace &workspace, const std::atomic<bool> *cancel) {
    validate_settings(settings);
    if (scene.scene.aggregates.count == 0u) {
        throw std::invalid_argument("CUDA ReSTIR scene is empty");
    }

    const std::uint32_t width = settings.frame.render.extent.width;
    const std::uint32_t height = settings.frame.render.extent.height;
    const std::uint32_t pixel_count = width * height;
    const bool gi_mode =
        settings.frame.render.integrator == IntegratorKind::ReSTIRGI;
    CudaRestirWorkspace::Impl &buffers = *workspace.m_impl;
    buffers.ensure_capacity(
        pixel_count,
        gi_mode ? RestirWorkspaceMode::GI : RestirWorkspaceMode::DI);
    RT_CUDA_CHECK(cudaMemset(
        buffers.film.data(), 0,
        static_cast<std::size_t>(pixel_count) * sizeof(CudaFilmPixel)));
    RT_CUDA_CHECK(cudaMemset(
        buffers.direct_film.data(), 0,
        static_cast<std::size_t>(pixel_count) * sizeof(CudaFilmPixel)));
    if (gi_mode) {
        RT_CUDA_CHECK(cudaMemset(
            buffers.indirect_film.data(), 0,
            static_cast<std::size_t>(pixel_count) *
                sizeof(CudaFilmPixel)));
    }
    RT_CUDA_CHECK(cudaMemset(buffers.counters.data(), 0,
                             sizeof(DeviceRestirCounters)));

    const restir::RestirFramePreparation preparation =
        restir::prepare_restir_frame(buffers.frame_state, settings.frame);
    const restir::RestirHistoryResetReason reset_reason =
        preparation.reset_reason;
    if (preparation.reset()) {
        buffers.committed_camera_valid = false;
    }
    SchedulerEvent begin;
    SchedulerEvent end;
    RT_CUDA_CHECK(cudaEventRecord(begin));
    std::uint64_t completed = 0;
    bool cancelled = false;
    auto cancel_at_pass_boundary = [&]() {
        if (cancel == nullptr ||
            !cancel->load(std::memory_order_relaxed)) {
            return false;
        }
        RT_CUDA_CHECK(cudaStreamSynchronize(nullptr));
        cancelled = true;
        return true;
    };
    for (std::uint32_t local_iteration = 0;
         local_iteration < settings.frame.render.samples_per_pixel;
         ++local_iteration) {
        if (cancel != nullptr &&
            cancel->load(std::memory_order_relaxed)) {
            cancelled = true;
            break;
        }
        if (buffers.frame_state.completed_iterations >
            std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error(
                "ReSTIR iteration seed index exceeds uint32 capacity");
        }
        const std::uint32_t iteration = static_cast<std::uint32_t>(
            buffers.frame_state.completed_iterations);
        const std::uint32_t write_gbuffer =
            buffers.frame_state.history_valid != 0u
                ? buffers.frame_state.committed_gbuffer ^ 1u
                : preparation.write_gbuffer;
        const std::uint32_t protected_reservoir =
            buffers.frame_state.di_history_valid != 0u
                ? buffers.frame_state.committed_di_reservoir
                : restir::kInvalidHistoryBuffer;
        const std::uint32_t write_reservoir =
            buffers.frame_state.di_history_valid != 0u
                ? restir::next_di_reservoir_buffer(
                      buffers.frame_state.committed_di_reservoir)
                : preparation.write_di_reservoir;
        const std::uint32_t protected_gi_reservoir =
            gi_mode && buffers.frame_state.gi_history_valid != 0u
                ? buffers.frame_state.committed_gi_reservoir
                : restir::kInvalidHistoryBuffer;
        const std::uint32_t write_gi_reservoir =
            gi_mode && buffers.frame_state.gi_history_valid != 0u
                ? restir::next_gi_reservoir_buffer(
                      buffers.frame_state.committed_gi_reservoir)
                : preparation.write_gi_reservoir;
        launch_restir_gbuffer(
            scene, width, height, iteration, settings.frame.render.seed,
            buffers.gbuffer[write_gbuffer].data(), buffers.counters.data(),
            settings.block_size);
        if (cancel_at_pass_boundary()) {
            break;
        }
        launch_restir_initial_di_candidates(
            scene, buffers.gbuffer[write_gbuffer].data(), width, height,
            iteration, settings.frame.render.seed,
            settings.frame.render.restir.initial_light_candidates,
            buffers.di_reservoir[write_reservoir].data(),
            buffers.counters.data(), settings.block_size);
        if (cancel_at_pass_boundary()) {
            break;
        }
        std::uint32_t final_reservoir = write_reservoir;
        if (settings.frame.render.restir.temporal_reuse) {
            const bool history_available =
                protected_reservoir != restir::kInvalidHistoryBuffer &&
                buffers.committed_camera_valid;
            const std::uint32_t destination_reservoir =
                select_di_scratch(final_reservoir, protected_reservoir);
            const std::uint32_t previous_gbuffer =
                history_available
                    ? buffers.frame_state.committed_gbuffer
                    : write_gbuffer;
            const std::uint32_t previous_reservoir =
                history_available ? protected_reservoir : final_reservoir;
            launch_restir_temporal_di(
                scene, buffers.gbuffer[write_gbuffer].data(),
                buffers.di_reservoir[final_reservoir].data(),
                buffers.gbuffer[previous_gbuffer].data(),
                buffers.di_reservoir[previous_reservoir].data(),
                buffers.committed_camera, history_available,
                buffers.di_reservoir[destination_reservoir].data(), width,
                height, iteration, settings.frame.render.seed,
                settings.frame.render.restir.max_history_length,
                settings.frame.render.restir.max_reservoir_candidates,
                settings.frame.render.restir.normal_threshold,
                settings.frame.render.restir.depth_threshold,
                settings.frame.render.restir.bias_correction ==
                    RestirBiasCorrection::Pairwise,
                buffers.counters.data(), settings.block_size);
            final_reservoir = destination_reservoir;
            if (cancel_at_pass_boundary()) {
                break;
            }
        }
        if (settings.frame.render.restir.spatial_reuse) {
            for (std::uint32_t pass = 0;
                 pass < settings.frame.render.restir.spatial_passes;
                 ++pass) {
                const std::uint32_t destination_reservoir =
                    select_di_scratch(final_reservoir,
                                      protected_reservoir);
                if (settings.frame.render.restir.bias_correction ==
                    RestirBiasCorrection::Pairwise) {
                    launch_restir_spatial_di_pairwise(
                        scene, buffers.gbuffer[write_gbuffer].data(),
                        buffers.di_reservoir[final_reservoir].data(),
                        buffers.di_reservoir[destination_reservoir].data(),
                        width, height, iteration, pass,
                        settings.frame.render.seed,
                        settings.frame.render.restir.spatial_neighbors,
                        settings.frame.render.restir.max_reservoir_candidates,
                        settings.frame.render.restir.normal_threshold,
                        settings.frame.render.restir.depth_threshold,
                        buffers.counters.data(), settings.block_size);
                } else {
                    launch_restir_spatial_di_basic(
                        scene, buffers.gbuffer[write_gbuffer].data(),
                        buffers.di_reservoir[final_reservoir].data(),
                        buffers.di_reservoir[destination_reservoir].data(),
                        width, height, iteration, pass,
                        settings.frame.render.seed,
                        settings.frame.render.restir.spatial_neighbors,
                        settings.frame.render.restir.max_reservoir_candidates,
                        settings.frame.render.restir.normal_threshold,
                        settings.frame.render.restir.depth_threshold,
                        buffers.counters.data(), settings.block_size);
                }
                final_reservoir = destination_reservoir;
                if (cancel_at_pass_boundary()) {
                    break;
                }
            }
        }
        if (cancelled) {
            break;
        }
        std::uint32_t final_gi_reservoir = write_gi_reservoir;
        if (gi_mode) {
            launch_restir_initial_gi_candidates(
                scene, buffers.gbuffer[write_gbuffer].data(), width,
                height, iteration, settings.frame.render.seed,
                settings.frame.render.restir.initial_gi_candidates,
                settings.reference_transport,
                buffers.gi_reservoir[write_gi_reservoir].data(),
                buffers.indirect_film.data(),
                buffers.counters.data(), settings.block_size);
            if (cancel_at_pass_boundary()) {
                break;
            }
            if (settings.frame.render.restir.temporal_reuse) {
                const bool history_available =
                    protected_gi_reservoir !=
                        restir::kInvalidHistoryBuffer &&
                    buffers.committed_camera_valid;
                const std::uint32_t destination = select_gi_scratch(
                    final_gi_reservoir, protected_gi_reservoir);
                const std::uint32_t previous_gbuffer =
                    history_available
                        ? buffers.frame_state.committed_gbuffer
                        : write_gbuffer;
                const std::uint32_t previous_reservoir =
                    history_available ? protected_gi_reservoir
                                      : final_gi_reservoir;
                launch_restir_temporal_gi_basic(
                    scene, buffers.gbuffer[write_gbuffer].data(),
                    buffers.gi_reservoir[final_gi_reservoir].data(),
                    buffers.gbuffer[previous_gbuffer].data(),
                    buffers.gi_reservoir[previous_reservoir].data(),
                    buffers.committed_camera, history_available,
                    buffers.gi_reservoir[destination].data(), width,
                    height, iteration, settings.frame.render.seed,
                    settings.frame.render.restir.max_history_length,
                    settings.frame.render.restir.max_reservoir_candidates,
                    settings.frame.render.restir.normal_threshold,
                    settings.frame.render.restir.depth_threshold,
                    buffers.counters.data(), settings.block_size);
                final_gi_reservoir = destination;
                if (cancel_at_pass_boundary()) {
                    break;
                }
            }
            if (settings.frame.render.restir.spatial_reuse) {
                for (std::uint32_t pass = 0u;
                     pass < settings.frame.render.restir.spatial_passes;
                     ++pass) {
                    const std::uint32_t destination = select_gi_scratch(
                        final_gi_reservoir, protected_gi_reservoir);
                    launch_restir_spatial_gi_basic(
                        scene, buffers.gbuffer[write_gbuffer].data(),
                        buffers.gi_reservoir[final_gi_reservoir].data(),
                        buffers.gi_reservoir[destination].data(), width,
                        height, iteration, pass,
                        settings.frame.render.seed,
                        settings.frame.render.restir.spatial_neighbors,
                        settings.frame.render.restir.max_reservoir_candidates,
                        settings.frame.render.restir.normal_threshold,
                        settings.frame.render.restir.depth_threshold,
                        buffers.counters.data(), settings.block_size);
                    final_gi_reservoir = destination;
                    if (cancel_at_pass_boundary()) {
                        break;
                    }
                }
            }
            if (cancelled) {
                break;
            }
        }
        launch_restir_initial_di_shading(
            scene, buffers.gbuffer[write_gbuffer].data(),
            buffers.di_reservoir[final_reservoir].data(), width, height,
            iteration, settings.frame.render.seed,
            static_cast<float>(settings.frame.render.sample_clamp),
            buffers.direct_film.data(), buffers.counters.data(),
            settings.block_size);
        if (gi_mode) {
            launch_restir_initial_gi_shading(
                scene, buffers.gbuffer[write_gbuffer].data(),
                buffers.gi_reservoir[final_gi_reservoir].data(), width,
                height, iteration, settings.frame.render.seed,
                buffers.indirect_film.data(), buffers.counters.data(),
                settings.block_size);
        }
        launch_restir_fallback_shading(
            scene, settings.reference_transport,
            buffers.gbuffer[write_gbuffer].data(), width, height, iteration,
            settings.frame.render.seed, buffers.direct_film.data(),
            buffers.counters.data(), settings.block_size);

        if (settings.generate_reference) {
            launch_restir_reference_shading(
                scene, settings.reference_transport, width, height,
                iteration, settings.frame.render.seed,
                static_cast<float>(settings.frame.render.sample_clamp),
                buffers.film.data(), buffers.counters.data(),
                settings.block_size);
        }
        RT_CUDA_CHECK(cudaStreamSynchronize(nullptr));
        restir::RestirHistoryCommit commit;
        commit.gbuffer = write_gbuffer;
        commit.di_reservoir = final_reservoir;
        if (gi_mode) {
            commit.gi_reservoir = final_gi_reservoir;
        }
        restir::commit_restir_iteration(
            buffers.frame_state, commit, settings.frame.frame_index);
        buffers.committed_camera = scene.scene.camera;
        buffers.committed_camera_valid = true;
        ++completed;
    }
    RT_CUDA_CHECK(cudaEventRecord(end));
    RT_CUDA_CHECK(cudaEventSynchronize(end));

    CudaRestirSchedulerOutput output;
    if (settings.generate_reference) {
        buffers.film.download_prefix(output.film, pixel_count);
    }
    buffers.direct_film.download_prefix(output.direct_film, pixel_count);
    if (gi_mode) {
        buffers.indirect_film.download_prefix(output.indirect_film,
                                              pixel_count);
    }
    if (buffers.frame_state.history_valid != 0u) {
        buffers.gbuffer[buffers.frame_state.committed_gbuffer]
            .download_prefix(output.gbuffer, pixel_count);
        buffers.di_reservoir[
            buffers.frame_state.committed_di_reservoir]
            .download_prefix(output.di_reservoirs, pixel_count);
        if (gi_mode && buffers.frame_state.gi_history_valid != 0u) {
            buffers.gi_reservoir[
                buffers.frame_state.committed_gi_reservoir]
                .download_prefix(output.gi_reservoirs, pixel_count);
        }
    }
    std::vector<DeviceRestirCounters> counters;
    buffers.counters.download_prefix(counters, 1u);
    const DeviceRestirCounters &counter = counters.front();
    RT_CUDA_CHECK(cudaEventElapsedTime(&output.stats.milliseconds, begin,
                                       end));
    output.stats.completed_iterations = completed;
    output.stats.sample_count =
        static_cast<std::uint64_t>(pixel_count) * completed;
    output.stats.traversal_steps = counter.traversal_steps;
    output.stats.shadow_rays = counter.shadow_rays;
    output.stats.clamped_samples = counter.clamped_samples;
    output.stats.invalid_samples = counter.invalid_samples;
    for (std::size_t index = 0; index < output.stats.gbuffer_status.size();
         ++index) {
        output.stats.gbuffer_status[index] =
            counter.gbuffer_status[index];
    }
    for (std::size_t index = 0;
         index < output.stats.transport_status.size(); ++index) {
        output.stats.transport_status[index] =
            counter.transport_status[index];
    }
    for (std::size_t index = 0;
         index < output.stats.di_generation_status.size(); ++index) {
        output.stats.di_generation_status[index] =
            counter.di_generation_status[index];
        output.stats.di_shading_status[index] =
            counter.di_shading_status[index];
        output.stats.di_spatial_status[index] =
            counter.di_spatial_status[index];
        output.stats.di_temporal_status[index] =
            counter.di_temporal_status[index];
    }
    for (std::size_t index = 0;
         index < output.stats.temporal_rejection.size(); ++index) {
        output.stats.temporal_rejection[index] =
            counter.temporal_rejection[index];
    }
    for (std::size_t index = 0;
         index < output.stats.spatial_compatibility.size(); ++index) {
        output.stats.spatial_compatibility[index] =
            counter.spatial_compatibility[index];
    }
    output.stats.initial_candidates = counter.initial_candidates;
    output.stats.represented_candidates =
        counter.represented_candidates;
    output.stats.rejected_candidates = counter.rejected_candidates;
    output.stats.spatial_candidates = counter.spatial_candidates;
    output.stats.spatial_accepted = counter.spatial_accepted;
    output.stats.spatial_rejected = counter.spatial_rejected;
    output.stats.pairwise_fallbacks = counter.pairwise_fallbacks;
    output.stats.temporal_candidates = counter.temporal_candidates;
    output.stats.temporal_accepted = counter.temporal_accepted;
    output.stats.temporal_pairwise_fallbacks =
        counter.temporal_pairwise_fallbacks;
    output.stats.valid_reservoirs = counter.valid_reservoirs;
    if (counter.valid_reservoirs != 0u) {
        const double inverse_valid =
            1.0 / static_cast<double>(counter.valid_reservoirs);
        output.stats.average_represented_M =
            static_cast<double>(counter.reservoir_M_sum) * inverse_valid;
        output.stats.average_effective_M =
            counter.reservoir_effective_M_sum * inverse_valid;
        output.stats.average_age =
            static_cast<double>(counter.reservoir_age_sum) * inverse_valid;
    }
    output.stats.visibility_rays = counter.visibility_rays;
    output.stats.di_clamped_samples = counter.di_clamped_samples;
    output.stats.di_invalid_samples = counter.di_invalid_samples;
    for (std::size_t index = 0u;
         index < output.stats.gi_generation_status.size(); ++index) {
        output.stats.gi_generation_status[index] =
            counter.gi_generation_status[index];
        output.stats.gi_shading_status[index] =
            counter.gi_shading_status[index];
        output.stats.gi_spatial_status[index] =
            counter.gi_spatial_status[index];
        output.stats.gi_temporal_status[index] =
            counter.gi_temporal_status[index];
        output.stats.gi_temporal_rejection[index] =
            counter.gi_temporal_rejection[index];
        output.stats.gi_shift_failures[index] =
            counter.gi_shift_failures[index];
    }
    for (std::size_t index = 0u;
         index < output.stats.gi_spatial_compatibility.size(); ++index) {
        output.stats.gi_spatial_compatibility[index] =
            counter.gi_spatial_compatibility[index];
    }
    output.stats.gi_initial_candidates = counter.gi_initial_candidates;
    output.stats.gi_represented_candidates =
        counter.gi_represented_candidates;
    output.stats.gi_rejected_candidates = counter.gi_rejected_candidates;
    output.stats.gi_spatial_candidates = counter.gi_spatial_candidates;
    output.stats.gi_spatial_accepted = counter.gi_spatial_accepted;
    output.stats.gi_temporal_candidates = counter.gi_temporal_candidates;
    output.stats.gi_temporal_accepted = counter.gi_temporal_accepted;
    output.stats.gi_valid_reservoirs = counter.gi_valid_reservoirs;
    if (counter.gi_valid_reservoirs != 0u) {
        const double inverse_valid =
            1.0 / static_cast<double>(counter.gi_valid_reservoirs);
        output.stats.gi_average_represented_M =
            static_cast<double>(counter.gi_reservoir_M_sum) *
            inverse_valid;
        output.stats.gi_average_effective_M =
            counter.gi_reservoir_effective_M_sum * inverse_valid;
        output.stats.gi_average_age =
            static_cast<double>(counter.gi_reservoir_age_sum) *
            inverse_valid;
    }
    output.stats.gi_visibility_rays = counter.gi_visibility_rays;
    output.stats.gi_fallbacks = counter.gi_fallbacks;
    output.stats.gi_invalid_samples = counter.gi_invalid_samples;
    output.stats.gi_suffix_shadow_rays = counter.gi_suffix_shadow_rays;
    output.stats.gi_suffix_traversal_steps =
        counter.gi_suffix_traversal_steps;
    output.stats.history_reset_reason = reset_reason;
    output.stats.workspace = workspace.info();
    output.stats.cancelled =
        cancelled || completed < settings.frame.render.samples_per_pixel;
    return output;
}

} // namespace cuda_backend

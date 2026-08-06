#include "restir_scheduler.h"

#include "cuda_error.h"
#include "restir_gbuffer.h"
#include "restir_initial_di.h"
#include "restir_reference_shading.h"
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
    if (!valid_integrator_policy(settings.reference_transport.policy) ||
        settings.reference_transport.max_depth == 0u) {
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
    if (!std::isfinite(settings.frame.render.sample_clamp) ||
        settings.frame.render.sample_clamp < 0.0) {
        throw std::invalid_argument("invalid ReSTIR sample clamp");
    }
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
    CudaRestirWorkspace::Impl &buffers = *workspace.m_impl;
    buffers.ensure_capacity(pixel_count);
    RT_CUDA_CHECK(cudaMemset(
        buffers.film.data(), 0,
        static_cast<std::size_t>(pixel_count) * sizeof(CudaFilmPixel)));
    RT_CUDA_CHECK(cudaMemset(
        buffers.direct_film.data(), 0,
        static_cast<std::size_t>(pixel_count) * sizeof(CudaFilmPixel)));
    RT_CUDA_CHECK(cudaMemset(buffers.counters.data(), 0,
                             sizeof(DeviceRestirCounters)));

    const restir::RestirFramePreparation preparation =
        restir::prepare_restir_frame(buffers.frame_state, settings.frame);
    const restir::RestirHistoryResetReason reset_reason =
        preparation.reset_reason;
    SchedulerEvent begin;
    SchedulerEvent end;
    RT_CUDA_CHECK(cudaEventRecord(begin));
    std::uint64_t completed = 0;
    bool cancelled = false;
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
        const std::uint32_t write_buffer =
            buffers.frame_state.history_valid != 0u
                ? buffers.frame_state.committed_buffer ^ 1u
                : preparation.write_buffer;
        launch_restir_gbuffer(
            scene, width, height, iteration, settings.frame.render.seed,
            buffers.gbuffer[write_buffer].data(), buffers.counters.data(),
            settings.block_size);
        launch_restir_initial_di_candidates(
            scene, buffers.gbuffer[write_buffer].data(), width, height,
            iteration, settings.frame.render.seed,
            settings.frame.render.restir.initial_light_candidates,
            buffers.di_reservoir[write_buffer].data(),
            buffers.counters.data(), settings.block_size);
        launch_restir_initial_di_shading(
            scene, buffers.gbuffer[write_buffer].data(),
            buffers.di_reservoir[write_buffer].data(), width, height,
            iteration, settings.frame.render.seed,
            static_cast<float>(settings.frame.render.sample_clamp),
            buffers.direct_film.data(), buffers.counters.data(),
            settings.block_size);

        launch_restir_reference_shading(
            scene, settings.reference_transport, width, height, iteration,
            settings.frame.render.seed,
            static_cast<float>(settings.frame.render.sample_clamp),
            buffers.film.data(), buffers.counters.data(),
            settings.block_size);
        RT_CUDA_CHECK(cudaStreamSynchronize(nullptr));
        restir::commit_restir_iteration(buffers.frame_state, write_buffer,
                                        settings.frame.frame_index);
        ++completed;
    }
    RT_CUDA_CHECK(cudaEventRecord(end));
    RT_CUDA_CHECK(cudaEventSynchronize(end));

    CudaRestirSchedulerOutput output;
    buffers.film.download_prefix(output.film, pixel_count);
    buffers.direct_film.download_prefix(output.direct_film, pixel_count);
    if (buffers.frame_state.history_valid != 0u) {
        buffers.gbuffer[buffers.frame_state.committed_buffer]
            .download_prefix(output.gbuffer, pixel_count);
        buffers.di_reservoir[buffers.frame_state.committed_buffer]
            .download_prefix(output.di_reservoirs, pixel_count);
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
    }
    output.stats.initial_candidates = counter.initial_candidates;
    output.stats.represented_candidates =
        counter.represented_candidates;
    output.stats.rejected_candidates = counter.rejected_candidates;
    output.stats.visibility_rays = counter.visibility_rays;
    output.stats.di_clamped_samples = counter.di_clamped_samples;
    output.stats.di_invalid_samples = counter.di_invalid_samples;
    output.stats.history_reset_reason = reset_reason;
    output.stats.workspace = workspace.info();
    output.stats.cancelled =
        cancelled || completed < settings.frame.render.samples_per_pixel;
    return output;
}

} // namespace cuda_backend

#ifndef CUDA_RESTIR_WORKSPACE_INTERNAL_H
#define CUDA_RESTIR_WORKSPACE_INTERNAL_H

#include "device_buffer.h"
#include "restir_device_types.h"
#include "restir_workspace.h"
#include "workspace_memory.h"

#include <algorithm>

namespace cuda_backend {

struct CudaRestirWorkspace::Impl {
    DeviceBuffer<restir::RestirSurface> gbuffer[2];
    DeviceBuffer<CudaFilmPixel> film;
    DeviceBuffer<DeviceRestirCounters> counters;
    restir::RestirFrameState frame_state;
    std::uint64_t allocation_generation = 0;

    std::size_t allocated_bytes() const noexcept {
        return gbuffer[0].bytes() + gbuffer[1].bytes() + film.bytes() +
               counters.bytes();
    }

    void ensure_capacity(std::uint32_t pixel_count) {
        const std::uint32_t pixel_capacity =
            std::max(pixel_count, film.size());
        const std::size_t required =
            static_cast<std::size_t>(pixel_capacity) *
                (2u * sizeof(restir::RestirSurface) +
                 sizeof(CudaFilmPixel)) +
            sizeof(DeviceRestirCounters);
        const std::size_t current = allocated_bytes();
        if (required > current) {
            ensure_cuda_workspace_fits(required, current,
                                       "CUDA ReSTIR workspace");
        }

        bool changed = false;
        changed |= gbuffer[0].ensure_capacity_discard(pixel_count);
        changed |= gbuffer[1].ensure_capacity_discard(pixel_count);
        changed |= film.ensure_capacity_discard(pixel_count);
        changed |= counters.ensure_capacity_discard(1u);
        if (changed) {
            ++allocation_generation;
            restir::reset_restir_history(frame_state);
        }
    }

    CudaRestirWorkspaceInfo info() const noexcept {
        CudaRestirWorkspaceInfo result;
        result.bytes = allocated_bytes();
        result.allocation_generation = allocation_generation;
        result.history_generation = frame_state.history_generation;
        result.completed_history_iterations =
            frame_state.completed_iterations;
        result.pixel_capacity = film.size();
        result.committed_buffer = frame_state.committed_buffer;
        result.history_valid = frame_state.history_valid != 0u;
        result.gbuffer_addresses[0] =
            reinterpret_cast<std::uintptr_t>(gbuffer[0].data());
        result.gbuffer_addresses[1] =
            reinterpret_cast<std::uintptr_t>(gbuffer[1].data());
        result.film_address =
            reinterpret_cast<std::uintptr_t>(film.data());
        return result;
    }
};

} // namespace cuda_backend

#endif

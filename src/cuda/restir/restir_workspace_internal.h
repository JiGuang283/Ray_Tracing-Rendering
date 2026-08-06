#ifndef CUDA_RESTIR_WORKSPACE_INTERNAL_H
#define CUDA_RESTIR_WORKSPACE_INTERNAL_H

#include "device_buffer.h"
#include "restir_device_types.h"
#include "restir_di_types.h"
#include "restir_workspace.h"
#include "workspace_memory.h"

#include <algorithm>

namespace cuda_backend {

struct CudaRestirWorkspace::Impl {
    DeviceBuffer<restir::RestirSurface> gbuffer[2];
    DeviceBuffer<restir::RestirDIReservoir> di_reservoir[2];
    DeviceBuffer<CudaFilmPixel> film;
    DeviceBuffer<CudaFilmPixel> direct_film;
    DeviceBuffer<DeviceRestirCounters> counters;
    restir::RestirFrameState frame_state;
    std::uint64_t allocation_generation = 0;

    std::size_t allocated_bytes() const noexcept {
        return gbuffer[0].bytes() + gbuffer[1].bytes() +
               di_reservoir[0].bytes() + di_reservoir[1].bytes() +
               film.bytes() + direct_film.bytes() + counters.bytes();
    }

    void ensure_capacity(std::uint32_t pixel_count) {
        const std::uint32_t pixel_capacity =
            std::max(pixel_count, film.size());
        const std::size_t required =
            static_cast<std::size_t>(pixel_capacity) *
                (2u * sizeof(restir::RestirSurface) +
                 2u * sizeof(restir::RestirDIReservoir) +
                 2u * sizeof(CudaFilmPixel)) +
            sizeof(DeviceRestirCounters);
        const std::size_t current = allocated_bytes();
        if (required > current) {
            ensure_cuda_workspace_fits(required, current,
                                       "CUDA ReSTIR workspace");
        }

        bool changed = false;
        changed |= gbuffer[0].ensure_capacity_discard(pixel_count);
        changed |= gbuffer[1].ensure_capacity_discard(pixel_count);
        changed |= di_reservoir[0].ensure_capacity_discard(pixel_count);
        changed |= di_reservoir[1].ensure_capacity_discard(pixel_count);
        changed |= film.ensure_capacity_discard(pixel_count);
        changed |= direct_film.ensure_capacity_discard(pixel_count);
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
        result.committed_gbuffer = frame_state.committed_gbuffer;
        result.committed_di_reservoir =
            frame_state.committed_di_reservoir;
        result.history_valid = frame_state.history_valid != 0u;
        result.gbuffer_addresses[0] =
            reinterpret_cast<std::uintptr_t>(gbuffer[0].data());
        result.gbuffer_addresses[1] =
            reinterpret_cast<std::uintptr_t>(gbuffer[1].data());
        result.reservoir_addresses[0] =
            reinterpret_cast<std::uintptr_t>(di_reservoir[0].data());
        result.reservoir_addresses[1] =
            reinterpret_cast<std::uintptr_t>(di_reservoir[1].data());
        result.film_address =
            reinterpret_cast<std::uintptr_t>(film.data());
        result.direct_film_address =
            reinterpret_cast<std::uintptr_t>(direct_film.data());
        return result;
    }
};

} // namespace cuda_backend

#endif

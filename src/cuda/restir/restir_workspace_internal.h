#ifndef CUDA_RESTIR_WORKSPACE_INTERNAL_H
#define CUDA_RESTIR_WORKSPACE_INTERNAL_H

#include "device_buffer.h"
#include "restir_device_types.h"
#include "restir_di_types.h"
#include "restir_gi_types.h"
#include "restir_workspace.h"
#include "packed_types.h"
#include "workspace_memory.h"

#include <algorithm>

namespace cuda_backend {

struct CudaRestirWorkspace::Impl {
    DeviceBuffer<restir::RestirSurface> gbuffer[2];
    DeviceBuffer<restir::RestirDIReservoir>
        di_reservoir[restir::kRestirDIReservoirBufferCount];
    DeviceBuffer<restir::RestirGIReservoir>
        gi_reservoir[restir::kRestirGIReservoirBufferCount];
    DeviceBuffer<CudaFilmPixel> film;
    DeviceBuffer<CudaFilmPixel> direct_film;
    DeviceBuffer<CudaFilmPixel> indirect_film;
    DeviceBuffer<DeviceRestirCounters> counters;
    restir::RestirFrameState frame_state;
    PackedCamera committed_camera{};
    bool committed_camera_valid = false;
    std::uint64_t allocation_generation = 0;
    RestirWorkspaceMode mode = RestirWorkspaceMode::DI;

    std::size_t allocated_bytes() const noexcept {
        return gbuffer[0].bytes() + gbuffer[1].bytes() +
               di_reservoir[0].bytes() + di_reservoir[1].bytes() +
               di_reservoir[2].bytes() +
               gi_reservoir[0].bytes() + gi_reservoir[1].bytes() +
               gi_reservoir[2].bytes() + film.bytes() +
               direct_film.bytes() + indirect_film.bytes() +
               counters.bytes();
    }

    void ensure_capacity(std::uint32_t pixel_count,
                         RestirWorkspaceMode requested_mode) {
        const std::uint32_t pixel_capacity =
            std::max(pixel_count, film.size());
        const bool needs_gi = requested_mode == RestirWorkspaceMode::GI;
        const std::size_t required =
            static_cast<std::size_t>(pixel_capacity) *
                (2u * sizeof(restir::RestirSurface) +
                 restir::kRestirDIReservoirBufferCount *
                     sizeof(restir::RestirDIReservoir) +
                 2u * sizeof(CudaFilmPixel) +
                 (needs_gi
                      ? restir::kRestirGIReservoirBufferCount *
                                sizeof(restir::RestirGIReservoir) +
                            sizeof(CudaFilmPixel)
                      : 0u)) +
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
        changed |= di_reservoir[2].ensure_capacity_discard(pixel_count);
        changed |= film.ensure_capacity_discard(pixel_count);
        changed |= direct_film.ensure_capacity_discard(pixel_count);
        if (needs_gi) {
            changed |= gi_reservoir[0].ensure_capacity_discard(pixel_count);
            changed |= gi_reservoir[1].ensure_capacity_discard(pixel_count);
            changed |= gi_reservoir[2].ensure_capacity_discard(pixel_count);
            changed |= indirect_film.ensure_capacity_discard(pixel_count);
        }
        changed |= counters.ensure_capacity_discard(1u);
        if (changed) {
            ++allocation_generation;
            restir::reset_restir_history(frame_state);
            committed_camera_valid = false;
        }
        mode = requested_mode;
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
        result.committed_gi_reservoir =
            frame_state.committed_gi_reservoir;
        result.mode = mode;
        result.history_valid = frame_state.history_valid != 0u;
        result.gbuffer_addresses[0] =
            reinterpret_cast<std::uintptr_t>(gbuffer[0].data());
        result.gbuffer_addresses[1] =
            reinterpret_cast<std::uintptr_t>(gbuffer[1].data());
        result.reservoir_addresses[0] =
            reinterpret_cast<std::uintptr_t>(di_reservoir[0].data());
        result.reservoir_addresses[1] =
            reinterpret_cast<std::uintptr_t>(di_reservoir[1].data());
        result.reservoir_addresses[2] =
            reinterpret_cast<std::uintptr_t>(di_reservoir[2].data());
        result.gi_reservoir_addresses[0] =
            reinterpret_cast<std::uintptr_t>(gi_reservoir[0].data());
        result.gi_reservoir_addresses[1] =
            reinterpret_cast<std::uintptr_t>(gi_reservoir[1].data());
        result.gi_reservoir_addresses[2] =
            reinterpret_cast<std::uintptr_t>(gi_reservoir[2].data());
        result.film_address =
            reinterpret_cast<std::uintptr_t>(film.data());
        result.direct_film_address =
            reinterpret_cast<std::uintptr_t>(direct_film.data());
        result.indirect_film_address =
            reinterpret_cast<std::uintptr_t>(indirect_film.data());
        return result;
    }
};

} // namespace cuda_backend

#endif

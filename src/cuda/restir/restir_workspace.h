#ifndef CUDA_RESTIR_WORKSPACE_H
#define CUDA_RESTIR_WORKSPACE_H

#include "restir_history.h"
#include "restir_surface.h"
#include "wavefront_renderer.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace cuda_backend {

struct CudaRestirWorkspaceInfo {
    std::size_t bytes = 0;
    std::uint64_t allocation_generation = 0;
    std::uint64_t history_generation = 0;
    std::uint64_t completed_history_iterations = 0;
    std::uint32_t pixel_capacity = 0;
    std::uint32_t committed_gbuffer = 0;
    std::uint32_t committed_di_reservoir = 0;
    bool history_valid = false;
    std::uintptr_t gbuffer_addresses[2]{};
    std::uintptr_t reservoir_addresses[2]{};
    std::uintptr_t film_address = 0;
    std::uintptr_t direct_film_address = 0;
};

struct CudaRestirSkeletonSettings;
struct CudaRestirSchedulerOutput;

class CudaRestirWorkspace {
  public:
    CudaRestirWorkspace();
    ~CudaRestirWorkspace();

    CudaRestirWorkspace(const CudaRestirWorkspace &) = delete;
    CudaRestirWorkspace &operator=(const CudaRestirWorkspace &) = delete;
    CudaRestirWorkspace(CudaRestirWorkspace &&) noexcept;
    CudaRestirWorkspace &operator=(CudaRestirWorkspace &&) noexcept;

    CudaRestirWorkspaceInfo info() const noexcept;
    void reset_history() noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;

    friend CudaRestirSchedulerOutput render_restir_skeleton_cuda(
        DeviceSceneView scene,
        const CudaRestirSkeletonSettings &settings,
        CudaRestirWorkspace &workspace,
        const std::atomic<bool> *cancel);
};

} // namespace cuda_backend

#endif

#include "restir_workspace_internal.h"

namespace cuda_backend {

CudaRestirWorkspace::CudaRestirWorkspace()
    : m_impl(std::make_unique<Impl>()) {
}

CudaRestirWorkspace::~CudaRestirWorkspace() = default;
CudaRestirWorkspace::CudaRestirWorkspace(CudaRestirWorkspace &&) noexcept =
    default;
CudaRestirWorkspace &CudaRestirWorkspace::operator=(
    CudaRestirWorkspace &&) noexcept = default;

CudaRestirWorkspaceInfo CudaRestirWorkspace::info() const noexcept {
    return m_impl ? m_impl->info() : CudaRestirWorkspaceInfo{};
}

void CudaRestirWorkspace::reset_history() noexcept {
    if (m_impl) {
        restir::reset_restir_history(m_impl->frame_state);
    }
}

} // namespace cuda_backend

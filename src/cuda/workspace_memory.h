#ifndef CUDA_WORKSPACE_MEMORY_H
#define CUDA_WORKSPACE_MEMORY_H

#include <cstddef>

namespace cuda_backend {

constexpr std::size_t kCudaWorkspaceMemoryReserve =
    64ull * 1024ull * 1024ull;

void ensure_cuda_workspace_fits(std::size_t required_bytes,
                                std::size_t reusable_bytes,
                                const char *workspace_name);

} // namespace cuda_backend

#endif

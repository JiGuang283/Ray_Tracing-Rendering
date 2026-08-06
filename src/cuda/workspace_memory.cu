#include "workspace_memory.h"

#include "device_scene.h"

#include <limits>
#include <sstream>
#include <stdexcept>

namespace cuda_backend {

void ensure_cuda_workspace_fits(std::size_t required_bytes,
                                std::size_t reusable_bytes,
                                const char *workspace_name) {
    const DeviceMemoryInfo memory = query_device_memory();
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    const std::size_t available =
        reusable_bytes > maximum - memory.free_bytes
            ? maximum
            : memory.free_bytes + reusable_bytes;
    const std::size_t usable = available > kCudaWorkspaceMemoryReserve
                                   ? available - kCudaWorkspaceMemoryReserve
                                   : 0;
    if (required_bytes <= usable) {
        return;
    }
    std::ostringstream message;
    message << workspace_name << " requires " << required_bytes
            << " bytes, but only " << memory.free_bytes
            << " device bytes are free after reserving "
            << kCudaWorkspaceMemoryReserve << " bytes";
    throw std::runtime_error(message.str());
}

} // namespace cuda_backend

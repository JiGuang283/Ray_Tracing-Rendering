#ifndef CUDA_RESTIR_DEVICE_TYPES_H
#define CUDA_RESTIR_DEVICE_TYPES_H

#include <type_traits>

namespace cuda_backend {

struct DeviceRestirCounters {
    unsigned long long gbuffer_status[7]{};
    unsigned long long transport_status[8]{};
    unsigned long long traversal_steps = 0;
    unsigned long long shadow_rays = 0;
    unsigned long long clamped_samples = 0;
    unsigned long long invalid_samples = 0;
};

static_assert(std::is_trivially_copyable_v<DeviceRestirCounters>);

} // namespace cuda_backend

#endif

#ifndef CUDA_RESTIR_DEVICE_TYPES_H
#define CUDA_RESTIR_DEVICE_TYPES_H

#include <type_traits>

namespace cuda_backend {

struct DeviceRestirCounters {
    unsigned long long gbuffer_status[7]{};
    unsigned long long transport_status[8]{};
    unsigned long long di_generation_status[11]{};
    unsigned long long di_shading_status[11]{};
    unsigned long long di_spatial_status[11]{};
    unsigned long long di_temporal_status[11]{};
    unsigned long long gi_generation_status[16]{};
    unsigned long long gi_shading_status[16]{};
    unsigned long long gi_spatial_status[16]{};
    unsigned long long gi_temporal_status[16]{};
    unsigned long long gi_spatial_compatibility[9]{};
    unsigned long long gi_temporal_rejection[16]{};
    unsigned long long spatial_compatibility[9]{};
    unsigned long long temporal_rejection[16]{};
    unsigned long long initial_candidates = 0;
    unsigned long long represented_candidates = 0;
    unsigned long long rejected_candidates = 0;
    unsigned long long spatial_candidates = 0;
    unsigned long long spatial_accepted = 0;
    unsigned long long spatial_rejected = 0;
    unsigned long long pairwise_fallbacks = 0;
    unsigned long long temporal_candidates = 0;
    unsigned long long temporal_accepted = 0;
    unsigned long long temporal_pairwise_fallbacks = 0;
    unsigned long long valid_reservoirs = 0;
    unsigned long long reservoir_M_sum = 0;
    unsigned long long reservoir_age_sum = 0;
    double reservoir_effective_M_sum = 0.0;
    unsigned long long visibility_rays = 0;
    unsigned long long di_clamped_samples = 0;
    unsigned long long di_invalid_samples = 0;
    unsigned long long gi_initial_candidates = 0;
    unsigned long long gi_represented_candidates = 0;
    unsigned long long gi_rejected_candidates = 0;
    unsigned long long gi_spatial_candidates = 0;
    unsigned long long gi_spatial_accepted = 0;
    unsigned long long gi_temporal_candidates = 0;
    unsigned long long gi_temporal_accepted = 0;
    unsigned long long gi_valid_reservoirs = 0;
    unsigned long long gi_reservoir_M_sum = 0;
    unsigned long long gi_reservoir_age_sum = 0;
    double gi_reservoir_effective_M_sum = 0.0;
    unsigned long long gi_visibility_rays = 0;
    unsigned long long gi_fallbacks = 0;
    unsigned long long gi_invalid_samples = 0;
    unsigned long long gi_shift_failures[16]{};
    unsigned long long gi_suffix_shadow_rays = 0;
    unsigned long long gi_suffix_traversal_steps = 0;
    unsigned long long traversal_steps = 0;
    unsigned long long shadow_rays = 0;
    unsigned long long clamped_samples = 0;
    unsigned long long invalid_samples = 0;
};

static_assert(std::is_trivially_copyable_v<DeviceRestirCounters>);

} // namespace cuda_backend

#endif

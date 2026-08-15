#ifndef INTEGRATOR_POLICY_H
#define INTEGRATOR_POLICY_H

#include "host_device.h"

#include <cstdint>

enum class RenderBackend : std::uint32_t {
    CPU,
    CUDA
};

enum class IntegratorKind : std::uint32_t {
    Path = 0,
    RussianRoulette = 1,
    PBRPath = 2,
    DirectLighting = 3,
    MISPath = 4,
    ReSTIRDI = 5,
    ReSTIRGI = 6,
    ReSTIRPT = 7,
};

enum class IntegratorExecutionModel : std::uint32_t {
    WavefrontPath = 0,
    RestirFrame = 1,
};

enum IntegratorPolicyFlags : std::uint32_t {
    INTEGRATOR_POLICY_NONE = 0,
    INTEGRATOR_POLICY_DIRECT_LIGHTING = 1u << 0,
    INTEGRATOR_POLICY_MIS = 1u << 1,
    INTEGRATOR_POLICY_RUSSIAN_ROULETTE = 1u << 2
};

struct IntegratorPolicy {
    IntegratorKind kind = IntegratorKind::MISPath;
    std::uint32_t flags = INTEGRATOR_POLICY_DIRECT_LIGHTING |
                          INTEGRATOR_POLICY_MIS |
                          INTEGRATOR_POLICY_RUSSIAN_ROULETTE;
    std::uint32_t rr_start_depth = 3;
    float rr_min_survival = 0.05f;

    RT_HOST_DEVICE bool uses_direct_lighting() const noexcept {
        return (flags & INTEGRATOR_POLICY_DIRECT_LIGHTING) != 0;
    }
    RT_HOST_DEVICE bool uses_mis() const noexcept {
        return (flags & INTEGRATOR_POLICY_MIS) != 0;
    }
    RT_HOST_DEVICE bool uses_russian_roulette() const noexcept {
        return (flags & INTEGRATOR_POLICY_RUSSIAN_ROULETTE) != 0;
    }
};

RT_HOST_DEVICE inline bool
valid_integrator_policy(const IntegratorPolicy &policy) noexcept {
    constexpr std::uint32_t known_flags =
        INTEGRATOR_POLICY_DIRECT_LIGHTING | INTEGRATOR_POLICY_MIS |
        INTEGRATOR_POLICY_RUSSIAN_ROULETTE;
    return static_cast<std::uint32_t>(policy.kind) <=
               static_cast<std::uint32_t>(IntegratorKind::MISPath) &&
           (policy.flags & ~known_flags) == 0 &&
           (!policy.uses_mis() || policy.uses_direct_lighting()) &&
           policy.rr_min_survival >= 0.0f &&
           policy.rr_min_survival <= 0.95f;
}

struct IntegratorDescriptor {
    IntegratorKind kind = IntegratorKind::MISPath;
    int id = 4;
    const char *name = "mis_path";
    IntegratorPolicy policy;
    bool supports_cpu = true;
    bool supports_cuda = true;
    IntegratorExecutionModel execution_model =
        IntegratorExecutionModel::WavefrontPath;
};

IntegratorKind integrator_kind_from_id(int id);
int integrator_id(IntegratorKind kind) noexcept;
const IntegratorDescriptor &integrator_descriptor(IntegratorKind kind);
bool integrator_supported(IntegratorKind kind, RenderBackend backend);
IntegratorPolicy integrator_policy(IntegratorKind kind);

#endif

#ifndef CPU_PATH_INTEGRATOR_H
#define CPU_PATH_INTEGRATOR_H

#include "integrator.h"
#include "render_types.h"

#include <memory>

class CpuPathIntegrator final : public Integrator {
  public:
    explicit CpuPathIntegrator(IntegratorPolicy policy);
    explicit CpuPathIntegrator(IntegratorKind kind);

    void set_max_depth(int depth) override;

    color Li(const ray &r, const hittable &scene, const color &background,
             IntegratorContext &context) const override;

    const IntegratorPolicy &policy() const noexcept {
        return m_policy;
    }

  private:
    IntegratorPolicy m_policy;
    int m_max_depth = 50;
};

std::shared_ptr<Integrator> make_cpu_integrator(IntegratorKind kind);

#endif

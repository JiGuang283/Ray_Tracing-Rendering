#include "render_types.h"

#include "rng.h"

#include <cmath>
#include <limits>
#include <stdexcept>

IntegratorKind integrator_kind_from_id(int id) {
    if (id < 0 || id > static_cast<int>(IntegratorKind::ReSTIRPT)) {
        throw std::invalid_argument("integrator id must be in the range 0..7");
    }
    return static_cast<IntegratorKind>(id);
}

int integrator_id(IntegratorKind kind) noexcept {
    return static_cast<int>(kind);
}

const IntegratorDescriptor &integrator_descriptor(IntegratorKind kind) {
    static const std::array<IntegratorDescriptor, 8> descriptors{{
        {IntegratorKind::Path,
         0,
         "path",
         {IntegratorKind::Path, INTEGRATOR_POLICY_NONE, 3, 0.0f},
         true,
         true,
         IntegratorExecutionModel::WavefrontPath},
        {IntegratorKind::RussianRoulette,
         1,
         "russian_roulette",
         {IntegratorKind::RussianRoulette,
          INTEGRATOR_POLICY_RUSSIAN_ROULETTE, 3, 0.005f},
         true,
         true,
         IntegratorExecutionModel::WavefrontPath},
        {IntegratorKind::PBRPath,
         2,
         "pbr_path",
         {IntegratorKind::PBRPath, INTEGRATOR_POLICY_RUSSIAN_ROULETTE, 3,
          0.05f},
         true,
         true,
         IntegratorExecutionModel::WavefrontPath},
        {IntegratorKind::DirectLighting,
         3,
         "direct_lighting",
         {IntegratorKind::DirectLighting,
          INTEGRATOR_POLICY_DIRECT_LIGHTING |
              INTEGRATOR_POLICY_RUSSIAN_ROULETTE,
          3, 0.05f},
         true,
         true,
         IntegratorExecutionModel::WavefrontPath},
        {IntegratorKind::MISPath,
         4,
         "mis_path",
         {IntegratorKind::MISPath,
          INTEGRATOR_POLICY_DIRECT_LIGHTING | INTEGRATOR_POLICY_MIS |
              INTEGRATOR_POLICY_RUSSIAN_ROULETTE,
          3, 0.05f},
         true,
         true,
         IntegratorExecutionModel::WavefrontPath},
        {IntegratorKind::ReSTIRDI,
         5,
         "restir_di",
         {IntegratorKind::ReSTIRDI, INTEGRATOR_POLICY_NONE, 0, 0.0f},
         false,
         true,
         IntegratorExecutionModel::RestirFrame},
        {IntegratorKind::ReSTIRGI,
         6,
         "restir_gi",
         {IntegratorKind::ReSTIRGI, INTEGRATOR_POLICY_NONE, 0, 0.0f},
         false,
         true,
         IntegratorExecutionModel::RestirFrame},
        {IntegratorKind::ReSTIRPT,
         7,
         "restir_pt",
         {IntegratorKind::ReSTIRPT, INTEGRATOR_POLICY_NONE, 0, 0.0f},
         false,
         false,
         IntegratorExecutionModel::RestirFrame},
    }};
    const std::uint32_t index = static_cast<std::uint32_t>(kind);
    if (index >= descriptors.size()) {
        throw std::invalid_argument("invalid integrator kind");
    }
    return descriptors[index];
}

bool integrator_supported(IntegratorKind kind, RenderBackend backend) {
    const IntegratorDescriptor &descriptor = integrator_descriptor(kind);
    return backend == RenderBackend::CPU ? descriptor.supports_cpu
                                         : descriptor.supports_cuda;
}

IntegratorPolicy integrator_policy(IntegratorKind kind) {
    const IntegratorDescriptor &descriptor = integrator_descriptor(kind);
    if (descriptor.execution_model !=
        IntegratorExecutionModel::WavefrontPath) {
        throw std::invalid_argument(
            "ReSTIR integrators do not use IntegratorPolicy");
    }
    return descriptor.policy;
}

std::size_t ImageExtent::pixel_count() const {
    return static_cast<std::size_t>(width) * height;
}

ImageExtent make_image_extent(int width, int height) {
    if (width < 2 || height < 2) {
        throw std::invalid_argument("render width and height must be at least 2");
    }
    const auto unsigned_width = static_cast<std::uint64_t>(width);
    const auto unsigned_height = static_cast<std::uint64_t>(height);
    const std::uint64_t pixels = unsigned_width * unsigned_height;
    if (pixels > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("render image exceeds uint32 pixel capacity");
    }
    return {static_cast<std::uint32_t>(width),
            static_cast<std::uint32_t>(height)};
}

ImageExtent make_image_extent(int width, double aspect_ratio) {
    if (!std::isfinite(aspect_ratio) || aspect_ratio <= 0.0) {
        throw std::invalid_argument("camera aspect ratio must be positive and finite");
    }
    if (width < 2) {
        throw std::invalid_argument("render width must be at least 2");
    }
    const double height_value = static_cast<double>(width) / aspect_ratio;
    if (!std::isfinite(height_value) ||
        height_value > static_cast<double>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("render height is out of range");
    }
    return make_image_extent(width, static_cast<int>(height_value));
}

void validate_render_request(const RenderRequest &request) {
    make_image_extent(static_cast<int>(request.extent.width),
                      static_cast<int>(request.extent.height));
    const IntegratorDescriptor &descriptor =
        integrator_descriptor(request.integrator);
    if (descriptor.execution_model ==
        IntegratorExecutionModel::WavefrontPath) {
        (void)integrator_policy(request.integrator);
    } else {
        validate_restir_settings(request.restir);
    }
    if (request.samples_per_pixel == 0) {
        throw std::invalid_argument("samples per pixel must be positive");
    }
    if (request.max_depth == 0) {
        throw std::invalid_argument("maximum path depth must be positive");
    }
    const std::uint32_t restir_stats_level =
        static_cast<std::uint32_t>(request.cuda_restir_stats_level);
    if (restir_stats_level >
        static_cast<std::uint32_t>(CudaRestirStatsLevel::Full)) {
        throw std::invalid_argument("invalid CUDA ReSTIR stats level");
    }
    if (!std::isfinite(request.sample_clamp) || request.sample_clamp < 0.0) {
        throw std::invalid_argument("sample clamp must be finite and non-negative");
    }
    if (!std::isfinite(request.color_pipeline.exposure) ||
        !std::isfinite(request.color_pipeline.gamma) ||
        request.color_pipeline.gamma <= 0.0) {
        throw std::invalid_argument("invalid color pipeline settings");
    }
}

CancellationToken::CancellationToken(
    std::shared_ptr<std::atomic<bool>> state)
    : m_state(std::move(state)) {
}

bool CancellationToken::is_cancelled() const noexcept {
    return m_state != nullptr && m_state->load(std::memory_order_relaxed);
}

const std::atomic<bool> *CancellationToken::native_flag() const noexcept {
    return m_state.get();
}

CancellationSource::CancellationSource()
    : m_state(std::make_shared<std::atomic<bool>>(false)) {
}

CancellationToken CancellationSource::token() const {
    return CancellationToken(m_state);
}

void CancellationSource::cancel() const noexcept {
    m_state->store(true, std::memory_order_relaxed);
}

bool CancellationSource::is_cancelled() const noexcept {
    return m_state->load(std::memory_order_relaxed);
}

std::uint32_t render_sample_seed(std::uint32_t base_seed,
                                 std::uint32_t pixel_index,
                                 std::uint32_t sample_index) noexcept {
    const std::uint32_t pixel_seed = mix_seed(base_seed, pixel_index + 1u);
    return mix_seed(pixel_seed, sample_index + 1u);
}

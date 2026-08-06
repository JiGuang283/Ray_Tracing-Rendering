#include "cuda_restir_device_checks.h"

#include "compiled_scene.h"
#include "device_scene.h"
#include "restir/restir_scheduler.h"
#include "restir_gbuffer_host_check.h"
#include "scene_compiler.h"
#include "scene_ir.h"
#include "wavefront_renderer.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string mode = "reservoir";
    std::filesystem::path scene_file =
        "assets/scenes/scene_023_mis_comparison_scene.json";
    std::uint32_t width = 32;
    std::uint32_t height = 18;
    std::uint32_t spp = 2;
    std::uint32_t max_depth = 6;
    std::uint32_t seed = 123;
    RestirBiasCorrection bias = RestirBiasCorrection::Pairwise;
    bool spatial = false;
};

Options parse_options(int argc, char **argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto value = [&]() -> std::string {
            if (++index >= argc) {
                throw std::runtime_error(argument + " requires a value");
            }
            return argv[index];
        };
        if (argument == "--mode") {
            options.mode = value();
        } else if (argument == "--scene-file") {
            options.scene_file = value();
        } else if (argument == "--width") {
            options.width = static_cast<std::uint32_t>(std::stoul(value()));
        } else if (argument == "--height") {
            options.height =
                static_cast<std::uint32_t>(std::stoul(value()));
        } else if (argument == "--spp") {
            options.spp = static_cast<std::uint32_t>(std::stoul(value()));
        } else if (argument == "--max-depth") {
            options.max_depth =
                static_cast<std::uint32_t>(std::stoul(value()));
        } else if (argument == "--seed") {
            options.seed = static_cast<std::uint32_t>(std::stoul(value()));
        } else if (argument == "--bias") {
            const std::string name = value();
            if (name == "basic") {
                options.bias = RestirBiasCorrection::Basic;
            } else if (name == "pairwise") {
                options.bias = RestirBiasCorrection::Pairwise;
            } else {
                throw std::runtime_error("invalid ReSTIR bias: " + name);
            }
        } else if (argument == "--spatial") {
            options.spatial = true;
        } else if (argument == "--help" || argument == "-h") {
            std::cout
                << "Usage: cuda_restir_check "
                   "[--mode reservoir|gbuffer|spatial|statistics] "
                   "[--scene-file PATH] [--width N] [--height N] "
                   "[--spp N] [--max-depth N] [--seed N] "
                   "[--bias basic|pairwise] [--spatial]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + argument);
        }
    }
    if ((options.mode != "reservoir" && options.mode != "gbuffer" &&
         options.mode != "spatial" && options.mode != "statistics") ||
        options.width < 2u || options.height < 2u || options.spp == 0u ||
        options.max_depth == 0u ||
        ((options.mode == "gbuffer" || options.mode == "spatial") &&
         options.spp != 1u)) {
        throw std::runtime_error("invalid CUDA ReSTIR check settings");
    }
    return options;
}

double image_mean_luminance(
    const std::vector<cuda_backend::CudaFilmPixel> &film) {
    double total = 0.0;
    std::uint64_t samples = 0;
    for (const cuda_backend::CudaFilmPixel &pixel : film) {
        total += 0.2126 * pixel.radiance.x +
                 0.7152 * pixel.radiance.y +
                 0.0722 * pixel.radiance.z;
        samples += pixel.sample_count;
    }
    return samples != 0u ? total / static_cast<double>(samples) : 0.0;
}

double sample_variance(const std::vector<double> &values) {
    if (values.size() < 2u) {
        return 0.0;
    }
    double mean = 0.0;
    for (double value : values) {
        mean += value;
    }
    mean /= static_cast<double>(values.size());
    double variance = 0.0;
    for (double value : values) {
        const double difference = value - mean;
        variance += difference * difference;
    }
    return variance / static_cast<double>(values.size() - 1u);
}

bool near(float left, float right, float tolerance = 2e-5f) {
    const float scale =
        std::max(1.0f, std::max(std::abs(left), std::abs(right)));
    return std::abs(left - right) <= tolerance * scale;
}

bool check_reservoir() {
    const CudaReservoirDeviceCheckResult result =
        run_cuda_reservoir_device_check();
    std::cout << "CUDA_RESTIR_CHECK"
              << " mode=reservoir"
              << " trials=" << result.trials
              << " heavy_observed=" << result.heavy_observed
              << " tolerance=" << result.tolerance
              << " failures=" << result.failures
              << " algebra=" << (result.algebra_pass ? "pass" : "fail")
              << " result=" << (result.passed() ? "pass" : "fail")
              << '\n';
    return result.passed();
}

bool check_gbuffer(const Options &options) {
    const SceneIR ir = load_scene_ir_file(options.scene_file.string());
    const CompiledScene compiled = compile_scene(ir);
    const CompiledSceneView host_scene = make_scene_view(compiled);
    cuda_backend::DeviceSceneStorage device_scene;
    device_scene.upload(compiled);

    cuda_backend::CudaRestirSkeletonSettings settings;
    settings.frame.render.extent =
        make_image_extent(static_cast<int>(options.width),
                          static_cast<int>(options.height));
    settings.frame.render.integrator = IntegratorKind::ReSTIRDI;
    settings.frame.render.samples_per_pixel = options.spp;
    settings.frame.render.max_depth = options.max_depth;
    settings.frame.render.seed = options.seed;
    settings.frame.render.sample_clamp = 0.0;
    settings.frame.render.restir.history_mode = RestirHistoryMode::Reset;
    settings.frame.render.restir.initial_bsdf_candidates = 0u;
    settings.frame.render.restir.temporal_reuse = false;
    settings.frame.render.restir.spatial_reuse = false;
    settings.frame.camera = ir.camera;
    settings.reference_transport.policy =
        integrator_policy(IntegratorKind::MISPath);
    settings.reference_transport.max_depth = options.max_depth;

    cuda_backend::CudaRestirWorkspace workspace;
    const cuda_backend::CudaRestirSchedulerOutput skeleton =
        cuda_backend::render_restir_skeleton_cuda(
            device_scene.view(), settings, workspace);
    const cuda_backend::CudaRestirWorkspaceInfo first_info =
        workspace.info();

    cuda_backend::CudaRenderSettings wavefront_settings;
    wavefront_settings.transport = settings.reference_transport;
    wavefront_settings.width = options.width;
    wavefront_settings.height = options.height;
    wavefront_settings.samples_per_pixel = options.spp;
    wavefront_settings.seed = options.seed;
    wavefront_settings.batch_size = 17u;
    cuda_backend::CudaRenderWorkspace wavefront_workspace;
    const cuda_backend::CudaRenderOutput wavefront =
        cuda_backend::render_wavefront_cuda(
            device_scene.view(), wavefront_settings, wavefront_workspace);

    std::uint64_t film_errors = 0;
    for (std::size_t index = 0; index < skeleton.film.size(); ++index) {
        const cuda_backend::CudaFilmPixel &left = skeleton.film[index];
        const cuda_backend::CudaFilmPixel &right = wavefront.film[index];
        if (left.sample_count != right.sample_count ||
            !near(left.radiance.x, right.radiance.x) ||
            !near(left.radiance.y, right.radiance.y) ||
            !near(left.radiance.z, right.radiance.z)) {
            ++film_errors;
        }
    }

    const std::uint32_t final_iteration = options.spp - 1u;
    const std::uint64_t surface_errors = compare_restir_gbuffer_host(
        host_scene, options.width, options.height, final_iteration,
        options.seed, skeleton.gbuffer);
    const RestirDIHostCheckResult di_check =
        compare_restir_initial_di_host(
            host_scene, options.width, options.height, options.spp,
            settings.frame.render.restir.initial_light_candidates,
            options.seed, skeleton.gbuffer, skeleton.di_reservoirs,
            skeleton.direct_film);
    const bool di_stats_match =
        skeleton.stats.initial_candidates ==
            di_check.initial_candidates &&
        skeleton.stats.represented_candidates ==
            di_check.represented_candidates &&
        skeleton.stats.rejected_candidates ==
            di_check.rejected_candidates &&
        skeleton.stats.visibility_rays == di_check.visibility_rays &&
        skeleton.stats.di_generation_status ==
            di_check.generation_status &&
        skeleton.stats.di_shading_status == di_check.shading_status;

    std::uint64_t gbuffer_failures = 0;
    for (std::size_t index = 2;
         index < skeleton.stats.gbuffer_status.size(); ++index) {
        gbuffer_failures += skeleton.stats.gbuffer_status[index];
    }
    const bool counter_match =
        skeleton.stats.transport_status == wavefront.stats.status_counts &&
        skeleton.stats.traversal_steps == wavefront.stats.traversal_steps &&
        skeleton.stats.shadow_rays == wavefront.stats.shadow_rays &&
        skeleton.stats.invalid_samples == wavefront.stats.invalid_samples;
    const bool initial_history =
        skeleton.stats.history_reset_reason ==
            restir::RestirHistoryResetReason::Explicit &&
        first_info.history_valid &&
        first_info.completed_history_iterations == options.spp &&
        first_info.committed_gbuffer == ((options.spp - 1u) & 1u) &&
        first_info.committed_di_reservoir ==
            ((options.spp - 1u) & 1u);

    settings.frame.render.samples_per_pixel = 1u;
    settings.frame.render.restir.history_mode =
        RestirHistoryMode::Continue;
    settings.frame.frame_index = 1u;
    const cuda_backend::CudaRestirSchedulerOutput continued =
        cuda_backend::render_restir_skeleton_cuda(
            device_scene.view(), settings, workspace);
    const cuda_backend::CudaRestirWorkspaceInfo continued_info =
        workspace.info();
    const bool reused =
        continued.stats.history_reset_reason ==
            restir::RestirHistoryResetReason::None &&
        continued_info.allocation_generation ==
            first_info.allocation_generation &&
        continued_info.completed_history_iterations == options.spp + 1u &&
        continued_info.gbuffer_addresses[0] ==
            first_info.gbuffer_addresses[0] &&
        continued_info.gbuffer_addresses[1] ==
            first_info.gbuffer_addresses[1] &&
        continued_info.reservoir_addresses[0] ==
            first_info.reservoir_addresses[0] &&
        continued_info.reservoir_addresses[1] ==
            first_info.reservoir_addresses[1] &&
        continued_info.film_address == first_info.film_address &&
        continued_info.direct_film_address ==
            first_info.direct_film_address;

    std::atomic<bool> cancel{true};
    settings.frame.frame_index = 2u;
    const std::uint32_t committed_gbuffer_before_cancel =
        continued_info.committed_gbuffer;
    const std::uint32_t committed_reservoir_before_cancel =
        continued_info.committed_di_reservoir;
    const cuda_backend::CudaRestirSchedulerOutput cancelled =
        cuda_backend::render_restir_skeleton_cuda(
            device_scene.view(), settings, workspace, &cancel);
    const cuda_backend::CudaRestirWorkspaceInfo cancelled_info =
        workspace.info();
    const bool cancellation_safe =
        cancelled.stats.cancelled &&
        cancelled.stats.completed_iterations == 0u &&
        cancelled_info.completed_history_iterations == options.spp + 1u &&
        cancelled_info.committed_gbuffer ==
            committed_gbuffer_before_cancel &&
        cancelled_info.committed_di_reservoir ==
            committed_reservoir_before_cancel;

    const bool passed = film_errors == 0u && surface_errors == 0u &&
                        di_check.reservoir_errors == 0u &&
                        di_check.direct_film_errors == 0u &&
                        di_stats_match &&
                        gbuffer_failures == 0u && counter_match &&
                        initial_history && reused && cancellation_safe;
    std::cout << "CUDA_RESTIR_CHECK"
              << " mode=gbuffer"
              << " pixels=" << options.width * options.height
              << " spp=" << options.spp
              << " film_errors=" << film_errors
              << " surface_errors=" << surface_errors
              << " reservoir_errors=" << di_check.reservoir_errors
              << " direct_errors=" << di_check.direct_film_errors
              << " di_stats=" << (di_stats_match ? "pass" : "fail")
              << " candidates=" << skeleton.stats.initial_candidates
              << " visibility_rays=" << skeleton.stats.visibility_rays
              << " gbuffer_failures=" << gbuffer_failures
              << " counter_match=" << (counter_match ? 1 : 0)
              << " history=" << (initial_history ? "pass" : "fail")
              << " reused=" << (reused ? 1 : 0)
              << " cancellation=" << (cancellation_safe ? 1 : 0)
              << " workspace_bytes=" << first_info.bytes
              << " result=" << (passed ? "pass" : "fail") << '\n';
    return passed;
}

bool check_statistics(const Options &options) {
    const SceneIR ir = load_scene_ir_file(options.scene_file.string());
    const CompiledScene compiled = compile_scene(ir);
    cuda_backend::DeviceSceneStorage device_scene;
    device_scene.upload(compiled);

    cuda_backend::CudaRestirSkeletonSettings settings;
    settings.frame.render.extent =
        make_image_extent(static_cast<int>(options.width),
                          static_cast<int>(options.height));
    settings.frame.render.integrator = IntegratorKind::ReSTIRDI;
    settings.frame.render.samples_per_pixel = options.spp;
    settings.frame.render.max_depth = 1u;
    settings.frame.render.sample_clamp = 0.0;
    settings.frame.render.restir.history_mode = RestirHistoryMode::Reset;
    settings.frame.render.restir.initial_bsdf_candidates = 0u;
    settings.frame.render.restir.temporal_reuse = false;
    settings.frame.render.restir.spatial_reuse = options.spatial;
    settings.frame.render.restir.bias_correction = options.bias;
    settings.frame.camera = ir.camera;
    settings.reference_transport.policy =
        integrator_policy(IntegratorKind::DirectLighting);
    settings.reference_transport.max_depth = 1u;

    cuda_backend::CudaRenderSettings nee_settings;
    nee_settings.transport = settings.reference_transport;
    nee_settings.width = options.width;
    nee_settings.height = options.height;
    nee_settings.samples_per_pixel = options.spp;
    nee_settings.batch_size = options.width * options.height;

    cuda_backend::CudaRestirWorkspace restir_workspace;
    cuda_backend::CudaRenderWorkspace nee_workspace;
    std::vector<double> restir_means;
    std::vector<double> nee_means;
    std::uint64_t restir_visibility = 0;
    std::uint64_t nee_visibility = 0;
    std::uint64_t initial_candidates = 0;
    constexpr std::uint32_t kSeedCount = 8u;
    for (std::uint32_t sequence = 0; sequence < kSeedCount; ++sequence) {
        const std::uint32_t seed = options.seed + sequence * 977u;
        settings.frame.render.seed = seed;
        settings.frame.frame_index = sequence;
        const cuda_backend::CudaRestirSchedulerOutput restir_output =
            cuda_backend::render_restir_skeleton_cuda(
                device_scene.view(), settings, restir_workspace);
        nee_settings.seed = seed;
        const cuda_backend::CudaRenderOutput nee_output =
            cuda_backend::render_wavefront_cuda(
                device_scene.view(), nee_settings, nee_workspace);
        restir_means.push_back(
            image_mean_luminance(restir_output.direct_film));
        nee_means.push_back(image_mean_luminance(nee_output.film));
        restir_visibility += restir_output.stats.visibility_rays;
        nee_visibility += nee_output.stats.shadow_rays;
        initial_candidates += restir_output.stats.initial_candidates;
    }

    double restir_mean = 0.0;
    double nee_mean = 0.0;
    for (std::size_t index = 0; index < restir_means.size(); ++index) {
        restir_mean += restir_means[index];
        nee_mean += nee_means[index];
    }
    restir_mean /= restir_means.size();
    nee_mean /= nee_means.size();
    const double relative_error =
        std::abs(restir_mean - nee_mean) / std::max(1e-6, nee_mean);
    const double restir_variance = sample_variance(restir_means);
    const double nee_variance = sample_variance(nee_means);
    const bool passed = relative_error <= 0.08 &&
                        restir_variance <= nee_variance * 1.25 &&
                        restir_visibility <=
                            static_cast<double>(nee_visibility) * 1.25 &&
                        restir_visibility < initial_candidates;
    std::cout << "CUDA_RESTIR_CHECK"
              << " mode=statistics"
              << " spatial=" << (options.spatial ? 1 : 0)
              << " bias="
              << (options.bias == RestirBiasCorrection::Pairwise
                      ? "pairwise"
                      : "basic")
              << " seeds=" << kSeedCount
              << " spp=" << options.spp
              << " restir_mean=" << restir_mean
              << " nee_mean=" << nee_mean
              << " relative_error=" << relative_error
              << " restir_variance=" << restir_variance
              << " nee_variance=" << nee_variance
              << " restir_visibility=" << restir_visibility
              << " nee_visibility=" << nee_visibility
              << " initial_candidates=" << initial_candidates
              << " result=" << (passed ? "pass" : "fail") << '\n';
    return passed;
}

bool check_spatial(const Options &options) {
    const SceneIR ir = load_scene_ir_file(options.scene_file.string());
    const CompiledScene compiled = compile_scene(ir);
    const CompiledSceneView host_scene = make_scene_view(compiled);
    cuda_backend::DeviceSceneStorage device_scene;
    device_scene.upload(compiled);

    cuda_backend::CudaRestirSkeletonSettings settings;
    settings.frame.render.extent =
        make_image_extent(static_cast<int>(options.width),
                          static_cast<int>(options.height));
    settings.frame.render.integrator = IntegratorKind::ReSTIRDI;
    settings.frame.render.samples_per_pixel = 1u;
    settings.frame.render.max_depth = options.max_depth;
    settings.frame.render.seed = options.seed;
    settings.frame.render.sample_clamp = 0.0;
    settings.frame.render.restir.history_mode = RestirHistoryMode::Reset;
    settings.frame.render.restir.initial_bsdf_candidates = 0u;
    settings.frame.render.restir.temporal_reuse = false;
    settings.frame.render.restir.spatial_reuse = true;
    settings.frame.render.restir.spatial_neighbors = 5u;
    settings.frame.render.restir.spatial_passes = 1u;
    settings.frame.render.restir.max_reservoir_candidates = 32u;
    settings.frame.render.restir.bias_correction = options.bias;
    settings.frame.camera = ir.camera;
    settings.reference_transport.policy =
        integrator_policy(IntegratorKind::MISPath);
    settings.reference_transport.max_depth = options.max_depth;

    cuda_backend::CudaRestirWorkspace workspace;
    const cuda_backend::CudaRestirSchedulerOutput output =
        cuda_backend::render_restir_skeleton_cuda(
            device_scene.view(), settings, workspace);
    const RestirDISpatialHostCheckResult host =
        options.bias == RestirBiasCorrection::Pairwise
            ? compare_restir_spatial_di_pairwise_host(
                  host_scene, options.width, options.height, 0u,
                  settings.frame.render.restir.initial_light_candidates,
                  settings.frame.render.restir.spatial_neighbors,
                  settings.frame.render.restir.spatial_passes,
                  settings.frame.render.restir.max_reservoir_candidates,
                  settings.frame.render.restir.normal_threshold,
                  settings.frame.render.restir.depth_threshold, options.seed,
                  output.gbuffer, output.di_reservoirs,
                  output.direct_film)
            : compare_restir_spatial_di_basic_host(
                  host_scene, options.width, options.height, 0u,
                  settings.frame.render.restir.initial_light_candidates,
                  settings.frame.render.restir.spatial_neighbors,
                  settings.frame.render.restir.spatial_passes,
                  settings.frame.render.restir.max_reservoir_candidates,
                  settings.frame.render.restir.normal_threshold,
                  settings.frame.render.restir.depth_threshold, options.seed,
                  output.gbuffer, output.di_reservoirs,
                  output.direct_film);
    const bool stats_match =
        output.stats.spatial_candidates == host.spatial_candidates &&
        output.stats.spatial_accepted == host.spatial_accepted &&
        output.stats.spatial_rejected == host.spatial_rejected &&
        output.stats.visibility_rays == host.visibility_rays &&
        output.stats.di_spatial_status == host.spatial_status &&
        output.stats.di_shading_status == host.shading_status &&
        output.stats.spatial_compatibility == host.compatibility &&
        output.stats.pairwise_fallbacks == host.pairwise_fallbacks;
    std::uint64_t capped_errors = 0u;
    for (const restir::RestirDIReservoir &reservoir :
         output.di_reservoirs) {
        if (reservoir.M >
            settings.frame.render.restir.max_reservoir_candidates) {
            ++capped_errors;
        }
    }
    const cuda_backend::CudaRestirWorkspaceInfo info = workspace.info();
    const bool distinct_commits = info.committed_gbuffer == 0u &&
                                  info.committed_di_reservoir == 1u;
    const bool passed = host.reservoir_errors == 0u &&
                        host.direct_film_errors == 0u && stats_match &&
                        capped_errors == 0u && distinct_commits;
    std::cout << "CUDA_RESTIR_CHECK"
              << " mode=spatial"
              << " bias="
              << (options.bias == RestirBiasCorrection::Pairwise
                      ? "pairwise"
                      : "basic")
              << " pixels=" << options.width * options.height
              << " reservoir_errors=" << host.reservoir_errors
              << " direct_errors=" << host.direct_film_errors
              << " candidates=" << output.stats.spatial_candidates
              << " accepted=" << output.stats.spatial_accepted
              << " rejected=" << output.stats.spatial_rejected
              << " pairwise_fallbacks="
              << output.stats.pairwise_fallbacks
              << " valid_reservoirs=" << output.stats.valid_reservoirs
              << " average_M=" << output.stats.average_represented_M
              << " average_effective_M="
              << output.stats.average_effective_M
              << " capped_errors=" << capped_errors
              << " distinct_commits=" << (distinct_commits ? 1 : 0)
              << " stats=" << (stats_match ? "pass" : "fail")
              << " result=" << (passed ? "pass" : "fail") << '\n';
    return passed;
}

} // namespace

int main(int argc, char **argv) {
    try {
        std::string reason;
        if (!cuda_backend::cuda_device_available(&reason)) {
            std::cout << "CUDA_RESTIR_SKIP reason=" << reason << '\n';
            return 77;
        }
        const Options options = parse_options(argc, argv);
        const bool passed =
            options.mode == "reservoir"
                ? check_reservoir()
                : (options.mode == "gbuffer"
                       ? check_gbuffer(options)
                       : (options.mode == "spatial"
                              ? check_spatial(options)
                              : check_statistics(options)));
        return passed
                   ? 0
                   : 1;
    } catch (const std::exception &error) {
        std::cerr << "CUDA_RESTIR_ERROR message=" << error.what() << '\n';
        return 1;
    }
}

#include "benchmark_report.h"

#include "json.hpp"

#include <fstream>
#include <stdexcept>

namespace {

using json = nlohmann::json;

json restir_stats_json(const RestirStats &stats) {
    json result;
    result["iterations"] = stats.iterations;
    result["initial_candidates"] = stats.initial_candidates;
    result["temporal_candidates"] = stats.temporal_candidates;
    result["temporal_accepted"] = stats.temporal_accepted;
    result["spatial_candidates"] = stats.spatial_candidates;
    result["spatial_accepted"] = stats.spatial_accepted;
    result["visibility_rays"] = stats.visibility_rays;
    result["history_resets"] = stats.history_resets;
    result["invalid_reservoirs"] = stats.invalid_reservoirs;
    result["shift_success"] = stats.shift_success;
    result["gi_initial_candidates"] = stats.gi_initial_candidates;
    result["gi_temporal_candidates"] = stats.gi_temporal_candidates;
    result["gi_temporal_accepted"] = stats.gi_temporal_accepted;
    result["gi_spatial_candidates"] = stats.gi_spatial_candidates;
    result["gi_spatial_accepted"] = stats.gi_spatial_accepted;
    result["gi_visibility_rays"] = stats.gi_visibility_rays;
    result["gi_fallbacks"] = stats.gi_fallbacks;
    result["gi_replay_candidates"] = stats.gi_replay_candidates;
    result["gi_replay_evaluations"] = stats.gi_replay_evaluations;
    result["gi_replay_shadow_rays"] = stats.gi_replay_shadow_rays;
    result["gi_replay_traversal_steps"] = stats.gi_replay_traversal_steps;
    result["gi_reconnect_selections"] = stats.gi_reconnect_selections;
    result["gi_replay_selections"] = stats.gi_replay_selections;
    result["gi_clamped_samples"] = stats.gi_clamped_samples;
    result["gi_invalid_reservoirs"] = stats.gi_invalid_reservoirs;
    result["average_M"] = stats.average_M;
    result["average_age"] = stats.average_age;
    result["gi_average_M"] = stats.gi_average_M;
    result["gi_average_age"] = stats.gi_average_age;
    result["gi_unique_source_pixels"] = stats.gi_unique_source_pixels;
    result["gi_max_source_reuse"] = stats.gi_max_source_reuse;
    result["gi_average_source_reuse"] = stats.gi_average_source_reuse;
    result["shift_failures"] = stats.shift_failures;
    result["history_failures"] = stats.history_failures;
    return result;
}

json render_stats_json(const RenderStats &stats) {
    const bool cuda_backend = stats.base.backend == "cuda";
    json result;
    result["seconds"] = stats.base.seconds;
    result["compile_seconds"] = stats.base.compile_seconds;
    result["upload_seconds"] = stats.base.upload_seconds;
    result["device_seconds"] = stats.cuda.device_seconds;
    result["resolve_seconds"] = stats.base.resolve_seconds;
    result["width"] = stats.base.width;
    result["height"] = stats.base.height;
    result["samples_per_pixel"] = stats.base.samples_per_pixel;
    result["requested_samples"] = stats.base.requested_samples;
    result["completed_samples"] = stats.base.completed_samples;
    result["sample_count"] = stats.base.sample_count;
    result["seed"] = stats.base.seed;
    result["threads"] = stats.cpu.threads;
    result["clamped_samples"] = stats.base.clamped_samples;
    result["invalid_samples"] = stats.base.invalid_samples;
    result["backend"] = stats.base.backend;
    result["device_name"] = stats.cuda.device_name;
    result["scene_bytes"] = stats.base.scene_bytes;
    result["workspace_bytes"] = stats.cuda.workspace_bytes;
    result["workspace_generation"] = stats.cuda.workspace_generation;
    result["workspace_pixel_capacity"] =
        stats.cuda.workspace_pixel_capacity;
    result["workspace_path_capacity"] =
        stats.cuda.workspace_path_capacity;
    result["traversal_steps"] =
        cuda_backend ? stats.cuda.traversal_steps
                     : stats.cpu.traversal_steps;
    result["shadow_rays"] = cuda_backend ? stats.cuda.shadow_rays
                                         : stats.cpu.shadow_rays;
    result["wavefront_advance_launches"] =
        stats.cuda.wavefront_advance_launches;
    result["wavefront_active_path_steps"] =
        stats.cuda.wavefront_active_path_steps;
    result["batch_size"] = stats.cuda.batch_size;
    result["batch_count"] = stats.cuda.batch_count;
    result["samples_per_launch"] = stats.cuda.samples_per_launch;
    result["status_counts"] = stats.cuda.status_counts;
    result["cancelled"] = stats.base.cancelled;
    result["restir"] = restir_stats_json(stats.restir);
    return result;
}

RestirStats accumulate_restir(const std::vector<RenderStats> &runs) {
    RestirStats totals;
    for (const RenderStats &stats : runs) {
        const RestirStats &r = stats.restir;
        totals.iterations += r.iterations;
        totals.initial_candidates += r.initial_candidates;
        totals.temporal_candidates += r.temporal_candidates;
        totals.temporal_accepted += r.temporal_accepted;
        totals.spatial_candidates += r.spatial_candidates;
        totals.spatial_accepted += r.spatial_accepted;
        totals.visibility_rays += r.visibility_rays;
        totals.history_resets += r.history_resets;
        totals.invalid_reservoirs += r.invalid_reservoirs;
        totals.shift_success += r.shift_success;
        totals.gi_initial_candidates += r.gi_initial_candidates;
        totals.gi_temporal_candidates += r.gi_temporal_candidates;
        totals.gi_temporal_accepted += r.gi_temporal_accepted;
        totals.gi_spatial_candidates += r.gi_spatial_candidates;
        totals.gi_spatial_accepted += r.gi_spatial_accepted;
        totals.gi_visibility_rays += r.gi_visibility_rays;
        totals.gi_fallbacks += r.gi_fallbacks;
        totals.gi_replay_candidates += r.gi_replay_candidates;
        totals.gi_replay_evaluations += r.gi_replay_evaluations;
        totals.gi_replay_shadow_rays += r.gi_replay_shadow_rays;
        totals.gi_replay_traversal_steps += r.gi_replay_traversal_steps;
        totals.gi_reconnect_selections += r.gi_reconnect_selections;
        totals.gi_replay_selections += r.gi_replay_selections;
        totals.gi_clamped_samples += r.gi_clamped_samples;
        totals.gi_invalid_reservoirs += r.gi_invalid_reservoirs;
        totals.average_M += r.average_M;
        totals.average_age += r.average_age;
        totals.gi_average_M += r.gi_average_M;
        totals.gi_average_age += r.gi_average_age;
        totals.gi_unique_source_pixels += r.gi_unique_source_pixels;
        totals.gi_max_source_reuse =
            std::max(totals.gi_max_source_reuse, r.gi_max_source_reuse);
        totals.gi_average_source_reuse += r.gi_average_source_reuse;
        for (std::size_t i = 0; i < totals.shift_failures.size(); ++i) {
            totals.shift_failures[i] += r.shift_failures[i];
        }
        for (std::size_t i = 0; i < totals.history_failures.size(); ++i) {
            totals.history_failures[i] += r.history_failures[i];
        }
    }
    return totals;
}

} // namespace

void write_benchmark_json_report(const std::string &path,
                                 const BenchmarkReportInput &input) {
    if (input.options == nullptr) {
        throw std::invalid_argument(
            "benchmark report requires parsed app options");
    }
    const AppOptions &options = *input.options;

    json root;
    root["schema_version"] = 1;
    root["generator"] = "CGAssignment4";
    root["options"] = {
        {"scene_id", options.scene_id},
        {"scene_file", options.scene_file},
        {"integrator_id", options.integrator_id},
        {"backend", options.render.backend == RenderBackend::CPU ? "cpu"
                                                                 : "cuda"},
        {"seed", options.render.seed},
        {"threads", options.render.threads},
        {"max_depth", options.render.max_depth},
        {"cuda_batch_size", options.render.cuda_batch_size},
        {"benchmark_runs", options.benchmark.runs},
    };
    root["preparation"] = {
        {"compile_seconds", input.preparation.compile_seconds},
        {"upload_seconds", input.preparation.upload_seconds},
        {"scene_bytes", input.preparation.scene_bytes},
    };

    json runs = json::array();
    std::uint64_t clamped_total = 0;
    std::uint64_t invalid_total = 0;
    for (std::size_t index = 0; index < input.runs.size(); ++index) {
        const RenderStats &stats = input.runs[index];
        const double samples_per_second =
            stats.base.seconds > 0.0
                ? static_cast<double>(stats.base.sample_count) /
                      stats.base.seconds
                : 0.0;
        runs.push_back({
            {"index", index + 1},
            {"samples_per_second", samples_per_second},
            {"stats", render_stats_json(stats)},
        });
        clamped_total += stats.base.clamped_samples;
        invalid_total += stats.base.invalid_samples;
    }
    root["runs"] = std::move(runs);

    root["summary"] = {
        {"runs", input.runs.size()},
        {"median_seconds", input.median_seconds},
        {"median_device_seconds", input.median_device_seconds},
        {"median_samples_per_second", input.median_samples_per_second},
        {"clamped_samples", clamped_total},
        {"invalid_samples", invalid_total},
        {"restir_totals", restir_stats_json(accumulate_restir(input.runs))},
    };

    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to open benchmark JSON output '" +
                                 path + "'");
    }
    output << root.dump(2) << '\n';
    if (!output) {
        throw std::runtime_error("failed to write benchmark JSON output '" +
                                 path + "'");
    }
}

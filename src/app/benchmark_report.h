#ifndef BENCHMARK_REPORT_H
#define BENCHMARK_REPORT_H

#include "app_options.h"

#include <string>
#include <vector>

struct BenchmarkReportInput {
    const AppOptions *options = nullptr;
    PreparationStats preparation;
    std::vector<RenderStats> runs;
    double median_seconds = 0.0;
    double median_device_seconds = 0.0;
    double median_samples_per_second = 0.0;
};

void write_benchmark_json_report(const std::string &path,
                                 const BenchmarkReportInput &input);

#endif

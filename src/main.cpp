#include "app_options.h"
#include "render_app.h"

#include <exception>
#include <iostream>

int main(int argc, char *args[]) {
    AppOptions options = parse_options(argc, args);
    if (!options.valid) {
        std::cerr << "Error: " << options.error << std::endl;
        print_usage();
        return 1;
    }

    try {
        if (options.benchmark.enabled) {
            return run_benchmark(options);
        }
        return run_windowed(options);
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return 1;
    }
}

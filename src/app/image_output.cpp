#include "image_output.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>

std::string save_rendered_image(const RenderBuffer &render_buffer, int scene_id,
                                int integrator_id) {
    mkdir("output", 0755);

    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
                         now.time_since_epoch())
                         .count();
    std::stringstream filename;
    filename << "output/scene" << std::setfill('0') << std::setw(2)
             << scene_id << "_integrator" << integrator_id << "_"
             << timestamp << ".png";

    std::string output_file = filename.str();
    if (render_buffer.save_to_png(output_file)) {
        std::cout << "Image saved successfully to " << output_file << std::endl;
    } else {
        std::cerr << "Failed to save image to " << output_file << std::endl;
    }
    return output_file;
}

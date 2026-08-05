#include "image_output.h"

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
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
    save_rendered_image_to(render_buffer, output_file);
    return output_file;
}

namespace {

void create_parent_directory(const std::string &path) {
    const std::filesystem::path parent =
        std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

} // namespace

void save_rendered_image_to(const RenderBuffer &render_buffer,
                            const std::string &path) {
    create_parent_directory(path);
    if (!render_buffer.save_to_png(path)) {
        throw std::runtime_error("failed to save image to '" + path + "'");
    }
    std::cout << "Image saved successfully to " << path << std::endl;
}

void save_linear_film_to(const BeautyFilm &film, const std::string &path) {
    create_parent_directory(path);
    film.save_to_pfm(path);
    std::cout << "Linear film saved successfully to " << path << std::endl;
}

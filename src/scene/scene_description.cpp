#include "scene_description.h"

#include "scene_ir.h"

#include <fstream>
#include <stdexcept>

SceneDescription load_scene_description(const std::string &path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Scene file error: cannot open '" + path +
                                 "'.");
    }

    SceneDescription description;
    description.source_path = path;
    input >> description.root;
    return description;
}

SceneIR load_scene_ir_file(const std::string &path) {
    return parse_scene_ir(load_scene_description(path));
}

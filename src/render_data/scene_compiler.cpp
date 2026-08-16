#include "scene_compiler.h"

#include "scene_compiler_internal.h"

using namespace scene_compiler_detail;

PackedCamera compile_packed_camera(const CameraConfig &camera, double time0,
                                   double time1) {
    validate_camera_config(camera);
    if (!std::isfinite(time0) || !std::isfinite(time1) || time1 < time0) {
        throw std::invalid_argument(
            "camera time interval must be finite and ordered");
    }
    return pack_camera(camera, time0, time1);
}

CompiledScene compile_scene(const SceneIR &ir) {
    CompiledScene scene =
        scene_compiler_detail::SceneCompiler(ir).compile();
    const ValidationReport validation = validate_compiled_scene(scene);
    if (!validation.ok()) {
        throw std::runtime_error("Compiled scene validation failed: " +
                                 validation.errors.front());
    }
    return scene;
}

CompiledScene load_compiled_scene(const std::string &path) {
    return compile_scene(load_scene_ir_file(path));
}

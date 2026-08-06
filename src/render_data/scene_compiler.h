#ifndef SCENE_COMPILER_H
#define SCENE_COMPILER_H

#include "compiled_scene.h"
#include "scene_ir.h"

#include <string>

PackedCamera compile_packed_camera(const CameraConfig &camera, double time0,
                                   double time1);
CompiledScene compile_scene(const SceneIR &ir);
CompiledScene load_compiled_scene(const std::string &path);

#endif

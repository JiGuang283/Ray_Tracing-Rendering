#ifndef SCENE_COMPILER_H
#define SCENE_COMPILER_H

#include "compiled_scene.h"
#include "scene_ir.h"

#include <string>

CompiledScene compile_scene(const SceneIR &ir);
CompiledScene load_compiled_scene(const std::string &path);

#endif

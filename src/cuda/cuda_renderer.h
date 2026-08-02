#ifndef CUDA_RENDERER_H
#define CUDA_RENDERER_H

#include "render_session.h"
#include "scene_ir.h"

#include <memory>

namespace cuda_backend {

std::unique_ptr<IRenderSession> make_cuda_render_session(const SceneIR &ir);

} // namespace cuda_backend

#endif

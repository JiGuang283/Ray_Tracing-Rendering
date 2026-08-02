#ifndef CUDA_RENDERER_H
#define CUDA_RENDERER_H

#include "compiled_scene.h"
#include "preview_surface.h"
#include "render_result.h"
#include "render_types.h"

namespace cuda_backend {

RenderResult render_cuda(const CompiledScene &scene,
                         const RenderRequest &request,
                         const CancellationToken &cancel = {},
                         PreviewSurface *preview = nullptr);

} // namespace cuda_backend

#endif

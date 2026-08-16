#ifndef CPU_PACKED_RENDER_SESSION_H
#define CPU_PACKED_RENDER_SESSION_H

#include "render_session.h"
#include "scene_ir.h"

#include <memory>

std::unique_ptr<IRenderSession>
make_cpu_packed_render_session(const SceneIR &ir);

#endif

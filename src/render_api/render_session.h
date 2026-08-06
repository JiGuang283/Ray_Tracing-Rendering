#ifndef RENDER_SESSION_H
#define RENDER_SESSION_H

#include "camera_config.h"
#include "preview_surface.h"
#include "render_result.h"

#include <cstdint>

struct SceneRevision {
    std::uint64_t camera = 0;
    std::uint64_t geometry = 0;
    std::uint64_t material = 0;
    std::uint64_t lighting = 0;
};

struct RenderFrameRequest {
    RenderRequest render;
    CameraConfig camera;
    std::uint64_t frame_index = 0;
    SceneRevision revision;
};

void validate_render_frame_request(const RenderFrameRequest &request);

class IRenderSession {
  public:
    virtual ~IRenderSession() = default;

    virtual const PreparationStats &preparation_stats() const noexcept = 0;
    virtual RenderResult render(const RenderRequest &request,
                                const CancellationToken &cancel,
                                PreviewSurface *preview = nullptr) = 0;
    virtual RenderResult render_frame(
        const RenderFrameRequest &request, const CancellationToken &cancel,
        PreviewSurface *preview = nullptr) = 0;
    virtual void reset_history() = 0;
};

#endif

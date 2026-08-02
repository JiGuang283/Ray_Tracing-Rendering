#ifndef RENDER_SESSION_H
#define RENDER_SESSION_H

#include "preview_surface.h"
#include "render_result.h"

class IRenderSession {
  public:
    virtual ~IRenderSession() = default;

    virtual const PreparationStats &preparation_stats() const noexcept = 0;
    virtual RenderResult render(const RenderRequest &request,
                                const CancellationToken &cancel,
                                PreviewSurface *preview = nullptr) = 0;
};

#endif

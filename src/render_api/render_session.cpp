#include "render_session.h"

void validate_render_frame_request(const RenderFrameRequest &request) {
    validate_render_request(request.render);
    validate_camera_config(request.camera);
}

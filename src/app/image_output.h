#ifndef IMAGE_OUTPUT_H
#define IMAGE_OUTPUT_H

#include "render_buffer.h"

#include <string>

std::string save_rendered_image(const RenderBuffer &render_buffer, int scene_id,
                                int integrator_id);

#endif

#ifndef IMAGE_OUTPUT_H
#define IMAGE_OUTPUT_H

#include "beauty_film.h"
#include "render_buffer.h"

#include <string>

std::string save_rendered_image(const RenderBuffer &render_buffer, int scene_id,
                                int integrator_id);
void save_rendered_image_to(const RenderBuffer &render_buffer,
                            const std::string &path);
void save_linear_film_to(const BeautyFilm &film, const std::string &path);

#endif

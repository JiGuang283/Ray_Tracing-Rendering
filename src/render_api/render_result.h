#ifndef RENDER_RESULT_H
#define RENDER_RESULT_H

#include "beauty_film.h"
#include "render_buffer.h"
#include "render_types.h"

struct RenderResult {
    BeautyFilm film;
    RenderBuffer display;
    RenderStats stats;
};

#endif

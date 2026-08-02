add_library(raytracer_render_api STATIC
    beauty_film.cpp
    color_pipeline.cpp
    preview_surface.cpp
    render_buffer.cpp
    render_types.cpp
)
add_library(raytracer::render_api ALIAS raytracer_render_api)

target_include_directories(raytracer_render_api
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${PROJECT_SOURCE_DIR}/src/external
)
target_link_libraries(raytracer_render_api PUBLIC
    raytracer::build_config
    raytracer::core
)

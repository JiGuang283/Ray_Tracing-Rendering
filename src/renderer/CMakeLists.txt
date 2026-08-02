add_library(raytracer_renderer STATIC
    color_pipeline.cpp
    direct_light_integrator.cpp
    film.cpp
    integrator_common.cpp
    mis_path_integrator.cpp
    path_integrator.cpp
    pbr_path_integrator.cpp
    render_buffer.cpp
    renderer.cpp
    rr_path_integrator.cpp
    sample_filter.cpp
)
add_library(raytracer::renderer ALIAS raytracer_renderer)

target_include_directories(raytracer_renderer
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${PROJECT_SOURCE_DIR}/src/external
)
target_link_libraries(raytracer_renderer PUBLIC
    raytracer::build_config
    raytracer::core
    raytracer::shading
    raytracer::materials
    raytracer::geometry
    raytracer::lighting
    Threads::Threads
)

add_library(raytracer_renderer STATIC
    cpu_path_integrator.cpp
    film.cpp
    integrator_common.cpp
    renderer.cpp
    sample_filter.cpp
)
add_library(raytracer::renderer ALIAS raytracer_renderer)

target_include_directories(raytracer_renderer
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(raytracer_renderer PUBLIC
    raytracer::build_config
    raytracer::core
    raytracer::render_api
    raytracer::shading
    raytracer::materials
    raytracer::geometry
    raytracer::lighting
    Threads::Threads
)

raytracer_enable_warnings(raytracer_renderer)

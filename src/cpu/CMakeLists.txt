add_library(raytracer_cpu_backend STATIC
    cpu_render_session.cpp
)
add_library(raytracer::cpu_backend ALIAS raytracer_cpu_backend)

target_include_directories(raytracer_cpu_backend PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(raytracer_cpu_backend PUBLIC
    raytracer::build_config
    raytracer::render_api
    raytracer::renderer
    raytracer::scene_runtime
)

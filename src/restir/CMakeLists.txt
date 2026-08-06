add_library(raytracer_restir STATIC
    restir_history.cpp
    restir_reference.cpp
    restir_validation.cpp
)
add_library(raytracer::restir ALIAS raytracer_restir)

target_include_directories(raytracer_restir PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(raytracer_restir PUBLIC
    raytracer::build_config
    raytracer::core
    raytracer::render_api
    raytracer::render_data
)

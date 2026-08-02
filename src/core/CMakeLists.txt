add_library(raytracer_core STATIC
    matrix4.cpp
    transform.cpp
)
add_library(raytracer::core ALIAS raytracer_core)

target_include_directories(raytracer_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(raytracer_core PUBLIC
    raytracer::build_config
)

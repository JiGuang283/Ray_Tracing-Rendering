add_library(raytracer_shading STATIC
    bsdf.cpp
    bsdf_closures.cpp
)
add_library(raytracer::shading ALIAS raytracer_shading)

target_include_directories(raytracer_shading PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(raytracer_shading PUBLIC
    raytracer::build_config
    raytracer::core
)

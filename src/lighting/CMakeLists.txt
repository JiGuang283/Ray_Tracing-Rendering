add_library(raytracer_lighting STATIC
    light_sampler.cpp
    material_emission.cpp
    mesh_light.cpp
)
add_library(raytracer::lighting ALIAS raytracer_lighting)

target_include_directories(raytracer_lighting PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(raytracer_lighting PUBLIC
    raytracer::build_config
    raytracer::core
    raytracer::shading
    raytracer::materials
    raytracer::geometry
)

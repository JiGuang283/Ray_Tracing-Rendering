add_library(raytracer_render_data STATIC
    compiled_scene.cpp
    flat_intersector.cpp
    packed_bsdf.cpp
    packed_bvh.cpp
    packed_light.cpp
    packed_material.cpp
    packed_texture.cpp
    packed_transport.cpp
    resource_compiler.cpp
    scene_compiler.cpp
)
add_library(raytracer::render_data ALIAS raytracer_render_data)

target_include_directories(raytracer_render_data PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(raytracer_render_data PUBLIC
    raytracer::build_config
    raytracer::core
    raytracer::materials
    raytracer::geometry
    raytracer::resources
    raytracer::scene
)

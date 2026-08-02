add_library(raytracer_scene STATIC
    scene_ir.cpp
    scene_light_loader.cpp
    scene_loader.cpp
    scene_loader_internal.cpp
    scene_material_loader.cpp
    scene_object_loader.cpp
    scenes.cpp
)
add_library(raytracer::scene ALIAS raytracer_scene)

target_include_directories(raytracer_scene PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src/external
)
target_link_libraries(raytracer_scene PUBLIC
    raytracer::build_config
    raytracer::core
    raytracer::materials
    raytracer::geometry
    raytracer::resources
    raytracer::lighting
    raytracer::renderer
)

add_library(raytracer_scene_ir STATIC
    scene_description.cpp
    scene_ir.cpp
    scene_json.cpp
)
add_library(raytracer::scene_ir ALIAS raytracer_scene_ir)

target_include_directories(raytracer_scene_ir PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src/external
)
target_link_libraries(raytracer_scene_ir PUBLIC
    raytracer::build_config
    raytracer::core
    raytracer::render_api
)

add_library(raytracer_scene_resources STATIC
    scene_material_loader.cpp
)
add_library(raytracer::scene_resources ALIAS raytracer_scene_resources)

target_include_directories(raytracer_scene_resources PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(raytracer_scene_resources PUBLIC
    raytracer::build_config
    raytracer::core
    raytracer::materials
    raytracer::resources
    raytracer::scene_ir
)

add_library(raytracer_scene_runtime STATIC
    scene_light_loader.cpp
    scene_loader.cpp
    scene_object_loader.cpp
    scenes.cpp
)
add_library(raytracer::scene_runtime ALIAS raytracer_scene_runtime)
add_library(raytracer::scene ALIAS raytracer_scene_runtime)

target_include_directories(raytracer_scene_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src/external
)
target_link_libraries(raytracer_scene_runtime PUBLIC
    raytracer::build_config
    raytracer::core
    raytracer::scene_ir
    raytracer::scene_resources
    raytracer::materials
    raytracer::geometry
    raytracer::resources
    raytracer::lighting
)

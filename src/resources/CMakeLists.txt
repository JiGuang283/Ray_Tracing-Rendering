add_library(raytracer_mikktspace STATIC
    ${PROJECT_SOURCE_DIR}/src/external/mikktspace/mikktspace.c
)
target_include_directories(raytracer_mikktspace PUBLIC
    ${PROJECT_SOURCE_DIR}/src/external/mikktspace
)

add_library(raytracer_resources STATIC
    asset_path.cpp
    gltf_importer.cpp
    obj_importer.cpp
    resource_registry.cpp
    ${PROJECT_SOURCE_DIR}/src/external/cgltf.cpp
)
add_library(raytracer::resources ALIAS raytracer_resources)

target_include_directories(raytracer_resources
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${PROJECT_SOURCE_DIR}/src/external
)
target_link_libraries(raytracer_resources
    PUBLIC
        raytracer::build_config
        raytracer::core
        raytracer::materials
        raytracer::geometry
    PRIVATE
        raytracer_mikktspace
)

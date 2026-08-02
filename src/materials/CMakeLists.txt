add_library(raytracer_materials STATIC
    image_asset.cpp
    material.cpp
    material_programs.cpp
    normal_mapping.cpp
    texture.cpp
    ${PROJECT_SOURCE_DIR}/src/external/stb_image.cpp
)
add_library(raytracer::materials ALIAS raytracer_materials)

target_include_directories(raytracer_materials
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${PROJECT_SOURCE_DIR}/src/external
)
target_link_libraries(raytracer_materials PUBLIC
    raytracer::build_config
    raytracer::core
    raytracer::shading
)

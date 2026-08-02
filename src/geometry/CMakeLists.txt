add_library(raytracer_geometry STATIC
    aarect.cpp
    accel.cpp
    box.cpp
    bvh.cpp
    hittable_list.cpp
    mesh_asset.cpp
    mesh_instance.cpp
    model_asset.cpp
    sphere.cpp
    transformed_hittable.cpp
)
add_library(raytracer::geometry ALIAS raytracer_geometry)

target_include_directories(raytracer_geometry PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(raytracer_geometry PUBLIC
    raytracer::build_config
    raytracer::core
    raytracer::shading
    raytracer::materials
)

add_library(raytracer_cpu_backend STATIC
    cpu_packed_render_session.cpp
    cpu_render_session.cpp
)
add_library(raytracer::cpu_backend ALIAS raytracer_cpu_backend)

target_include_directories(raytracer_cpu_backend PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(raytracer_cpu_backend PUBLIC
    raytracer::build_config
    raytracer::render_api
    raytracer::render_data
    raytracer::renderer
    raytracer::scene_runtime
)

option(RAYTRACER_PACKED_HOST_FAST
       "Enable fast-math and x86-64-v3 tuning for the packed CPU backend" ON)
if(RAYTRACER_PACKED_HOST_FAST)
    set_source_files_properties(cpu_packed_render_session.cpp PROPERTIES
        COMPILE_OPTIONS "-Ofast;-march=x86-64-v3;-funroll-loops"
    )
endif()

raytracer_enable_warnings(raytracer_cpu_backend)

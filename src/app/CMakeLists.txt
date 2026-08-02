add_executable(CGAssignment4
    ${PROJECT_SOURCE_DIR}/src/main.cpp
    WindowsApp.cpp
    app_options.cpp
    image_output.cpp
    render_app.cpp
)
set_target_properties(CGAssignment4 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}
)

target_include_directories(CGAssignment4 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(CGAssignment4
    PRIVATE
        raytracer::build_config
        raytracer::scene
        raytracer::renderer
        SDL2::SDL2main
        SDL2::SDL2
)

if(RAYTRACER_ENABLE_CUDA)
    target_link_libraries(CGAssignment4 PRIVATE
        raytracer::render_data
        raytracer::cuda
    )
    target_compile_definitions(CGAssignment4 PRIVATE RAYTRACER_HAS_CUDA=1)
else()
    target_compile_definitions(CGAssignment4 PRIVATE RAYTRACER_HAS_CUDA=0)
endif()

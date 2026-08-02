function(raytracer_add_cpu_tool target)
    add_executable(${target} ${target}.cpp)
    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}
    )
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${target} PRIVATE raytracer::render_data)
endfunction()

raytracer_add_cpu_tool(scene_data_check)
raytracer_add_cpu_tool(scene_intersection_check)
raytracer_add_cpu_tool(packed_transport_check)

target_link_libraries(scene_intersection_check PRIVATE
    raytracer::scene_runtime
)

if(RAYTRACER_ENABLE_CUDA)
    function(raytracer_add_cuda_tool target)
        add_executable(${target} ${target}.cpp)
        set_target_properties(${target} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}
        )
        target_include_directories(${target} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
        )
        target_link_libraries(${target} PRIVATE raytracer::cuda)
    endfunction()

    raytracer_add_cuda_tool(cuda_scene_check)
    raytracer_add_cuda_tool(cuda_light_check)
    raytracer_add_cuda_tool(cuda_shading_check)
    raytracer_add_cuda_tool(cuda_transport_check)
endif()

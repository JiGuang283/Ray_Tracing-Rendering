function(raytracer_add_cpu_tool target)
    add_executable(${target} ${target}.cpp)
    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}
    )
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${target} PRIVATE raytracer::render_data)
    raytracer_enable_warnings(${target})
endfunction()

raytracer_add_cpu_tool(scene_data_check)
raytracer_add_cpu_tool(scene_intersection_check)
raytracer_add_cpu_tool(packed_transport_check)

add_executable(restir_reservoir_check restir_reservoir_check.cpp)
set_target_properties(restir_reservoir_check PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}
)
target_link_libraries(restir_reservoir_check PRIVATE raytracer::restir)

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
        raytracer_enable_warnings(${target})
    endfunction()

    raytracer_add_cuda_tool(cuda_scene_check)
    raytracer_add_cuda_tool(cuda_light_check)
    raytracer_add_cuda_tool(cuda_shading_check)
    raytracer_add_cuda_tool(cuda_transport_check)

    add_executable(cuda_restir_check
        cuda_restir_check.cpp
        cuda_restir_device_checks.cu
        restir_gbuffer_host_check.cpp
    )
    set_target_properties(cuda_restir_check PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
    )
    target_link_libraries(cuda_restir_check PRIVATE
        raytracer::cuda
    )
    target_compile_options(cuda_restir_check PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
    raytracer_enable_warnings(cuda_restir_check)
endif()

raytracer_enable_warnings(restir_reservoir_check)

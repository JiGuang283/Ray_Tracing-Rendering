add_executable(raytracer_tests
    test_bsdf.cpp
    test_compiled_scene.cpp
    test_core.cpp
    test_light_sampler.cpp
    test_main.cpp
    test_material.cpp
    test_mesh_asset.cpp
    test_model_asset.cpp
    test_packed_bsdf.cpp
    test_packed_light.cpp
    test_packed_shading.cpp
    test_packed_transport.cpp
    test_restir_reservoir.cpp
    test_render_runtime.cpp
    test_sample_filter.cpp
    test_scene_ir.cpp
    test_texture.cpp
    test_transform.cpp
    test_triangle_intersection.cpp
)
set_target_properties(raytracer_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}
)
target_include_directories(raytracer_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(raytracer_tests PRIVATE
    raytracer::core
    raytracer::render_api
    raytracer::shading
    raytracer::materials
    raytracer::geometry
    raytracer::resources
    raytracer::lighting
    raytracer::renderer
    raytracer::scene
    raytracer::render_data
    raytracer::restir
)

add_test(NAME raytracer_tests COMMAND raytracer_tests)
add_test(
    NAME packed_transport_check
    COMMAND packed_transport_check --ids 7,23 --width 8 --spp 1
            --max-depth 4 --integrator 4 --seed 123
)
set_tests_properties(packed_transport_check PROPERTIES
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

if(RAYTRACER_ENABLE_CUDA)
    add_test(NAME cuda_restir_reservoir_check COMMAND cuda_restir_check)
    add_test(
        NAME cuda_scene_check
        COMMAND cuda_scene_check --ids 1,7,23 --grid-width 9
                --grid-height 7
    )
    add_test(NAME cuda_light_check
        COMMAND cuda_light_check --ids 7,8,11,23
    )
    add_test(
        NAME cuda_shading_check
        COMMAND cuda_shading_check --ids 7,8,11,23 --grid-width 9
                --grid-height 7
    )
    foreach(integrator RANGE 0 4)
        add_test(
            NAME cuda_transport_policy_${integrator}
            COMMAND cuda_transport_check --ids 1,7,23 --width 8
                    --height 6 --spp 1 --max-depth 6 --batch-size 17
                    --integrator ${integrator}
        )
    endforeach()

    set_tests_properties(
        cuda_scene_check
        cuda_light_check
        cuda_shading_check
        cuda_transport_policy_0
        cuda_transport_policy_1
        cuda_transport_policy_2
        cuda_transport_policy_3
        cuda_transport_policy_4
        cuda_restir_reservoir_check
        PROPERTIES
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            SKIP_RETURN_CODE 77
    )
endif()

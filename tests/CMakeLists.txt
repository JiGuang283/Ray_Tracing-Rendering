add_executable(raytracer_tests
    test_bsdf.cpp
    test_compiled_scene.cpp
    test_core.cpp
    test_light_sampler.cpp
    test_main.cpp
    test_occluded.cpp
    test_material.cpp
    test_mesh_asset.cpp
    test_model_asset.cpp
    test_packed_bsdf.cpp
    test_packed_light.cpp
    test_packed_shading.cpp
    test_packed_transport.cpp
    test_restir_frame.cpp
    test_restir_reservoir.cpp
    test_render_revision.cpp
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
    raytracer::cpu_backend
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

if(TARGET CGAssignment4)
    add_test(
        NAME cpu_packed_session_check
        COMMAND $<TARGET_FILE:CGAssignment4> 23 4 --bench
                --cpu-packed --width 32 --spp 2 --max-depth 4
                --seed 123 --runs 1
    )
    set_tests_properties(cpu_packed_session_check PROPERTIES
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    )
endif()

if(RAYTRACER_ENABLE_CUDA)
    if(TARGET CGAssignment4)
        add_test(NAME cuda_restir_di_session
            COMMAND $<TARGET_FILE:CGAssignment4> 65 5 --bench
                    --backend cuda --width 32 --spp 2 --max-depth 4
                    --seed 123 --runs 1
        )
        add_test(NAME cuda_restir_gi_session
            COMMAND $<TARGET_FILE:CGAssignment4> 68 6 --bench
                    --backend cuda --width 32 --spp 2 --max-depth 4
                    --seed 123 --runs 1
        )
        set_tests_properties(
            cuda_restir_di_session
            cuda_restir_gi_session
            PROPERTIES
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            SKIP_REGULAR_EXPRESSION "CUDA backend unavailable"
        )
    endif()
    add_test(NAME cuda_restir_reservoir_check
        COMMAND cuda_restir_check --mode reservoir
    )
    add_test(NAME cuda_restir_gbuffer_check
        COMMAND cuda_restir_check --mode gbuffer --width 32 --height 18
                --spp 1 --max-depth 6 --seed 123
    )
    add_test(NAME cuda_restir_quad_check
        COMMAND cuda_restir_check --mode gbuffer
                --scene-file assets/scenes/scene_065_restir_many_light.json
                --width 32 --height 18 --spp 1 --max-depth 4 --seed 123
    )
    add_test(NAME cuda_restir_initial_gi_check
        COMMAND cuda_restir_check --mode gi
                --scene-file assets/scenes/scene_068_restir_diffuse_gi.json
                --width 32 --height 32 --spp 1 --max-depth 4 --seed 123
    )
    add_test(NAME cuda_restir_gi_reuse_check
        COMMAND cuda_restir_check --mode gi-reuse
                --scene-file assets/scenes/scene_068_restir_diffuse_gi.json
                --width 32 --height 32 --spp 1 --max-depth 4 --seed 123
                --bias pairwise
    )
    add_test(NAME cuda_restir_gi_reuse_basic_check
        COMMAND cuda_restir_check --mode gi-reuse
                --scene-file assets/scenes/scene_068_restir_diffuse_gi.json
                --width 32 --height 32 --spp 1 --max-depth 4 --seed 123
                --bias basic
    )
    add_test(NAME cuda_restir_gi_replay_check
        COMMAND cuda_restir_check --mode gi-reuse
                --scene-file assets/scenes/scene_069_restir_glossy_gi.json
                --width 32 --height 32 --spp 1 --max-depth 4 --seed 123
                --bias pairwise --require-replay
    )
    add_test(NAME cuda_restir_glossy_gi_initial_check
        COMMAND cuda_restir_check --mode gi
                --scene-file assets/scenes/scene_069_restir_glossy_gi.json
                --width 32 --height 32 --spp 1 --max-depth 4 --seed 123
    )
    add_test(NAME cuda_restir_gi_final_gather_check
        COMMAND cuda_restir_check --mode gi-statistics
                --scene-file assets/scenes/scene_068_restir_diffuse_gi.json
                --width 16 --height 16 --spp 8 --max-depth 4 --seed 123
                --spatial --temporal --bias pairwise --final-gather
    )
    add_test(NAME cuda_restir_initial_gi_statistics
        COMMAND cuda_restir_check --mode gi-statistics
                --scene-file assets/scenes/scene_068_restir_diffuse_gi.json
                --width 32 --height 32 --spp 32 --max-depth 4 --seed 123
    )
    add_test(NAME cuda_restir_basic_gi_statistics
        COMMAND cuda_restir_check --mode gi-statistics
                --scene-file assets/scenes/scene_068_restir_diffuse_gi.json
                --width 32 --height 32 --spp 32 --max-depth 4 --seed 123
                --spatial --temporal --bias basic
    )
    add_test(NAME cuda_restir_pairwise_gi_statistics
        COMMAND cuda_restir_check --mode gi-statistics
                --scene-file assets/scenes/scene_068_restir_diffuse_gi.json
                --width 32 --height 32 --spp 32 --max-depth 4 --seed 123
                --spatial --temporal --bias pairwise
    )
    add_test(NAME cuda_restir_glossy_gi_statistics
        COMMAND cuda_restir_check --mode gi-statistics
                --scene-file assets/scenes/scene_069_restir_glossy_gi.json
                --width 32 --height 32 --spp 2048 --max-depth 4 --seed 123
                --require-replay
    )
    add_test(NAME cuda_restir_spatial_pairwise_check
        COMMAND cuda_restir_check --mode spatial
                --scene-file assets/scenes/scene_065_restir_many_light.json
                --width 32 --height 18 --spp 1 --max-depth 4 --seed 123
                --bias pairwise
    )
    add_test(NAME cuda_restir_temporal_pairwise_check
        COMMAND cuda_restir_check --mode temporal
                --scene-file assets/scenes/scene_065_restir_many_light.json
                --width 32 --height 18 --spp 1 --max-depth 4 --seed 123
                --bias pairwise
    )
    add_test(NAME cuda_restir_environment_check
        COMMAND cuda_restir_check --mode gbuffer
                --scene-file assets/scenes/scene_019_environment_light_scene.json
                --width 32 --height 18 --spp 1 --max-depth 4 --seed 123
    )
    add_test(NAME cuda_restir_sphere_mesh_check
        COMMAND cuda_restir_check --mode gbuffer
                --scene-file assets/scenes/scene_005_simple_light.json
                --width 32 --height 18 --spp 1 --max-depth 4 --seed 123
    )
    add_test(NAME cuda_restir_delta_only_check
        COMMAND cuda_restir_check --mode gbuffer
                --scene-file assets/scenes/scene_015_point_light_scene.json
                --width 32 --height 18 --spp 1 --max-depth 4 --seed 123
    )
    add_test(NAME cuda_restir_initial_di_statistics
        COMMAND cuda_restir_check --mode statistics
                --scene-file assets/scenes/scene_065_restir_many_light.json
                --width 32 --height 18 --spp 64 --max-depth 1 --seed 123
    )
    add_test(NAME cuda_restir_spatial_di_statistics
        COMMAND cuda_restir_check --mode statistics --spatial
                --scene-file assets/scenes/scene_065_restir_many_light.json
                --width 32 --height 18 --spp 64 --max-depth 1 --seed 123
                --bias pairwise
    )
    add_test(NAME cuda_restir_spatiotemporal_di_statistics
        COMMAND cuda_restir_check --mode statistics --spatial --temporal
                --scene-file assets/scenes/scene_065_restir_many_light.json
                --width 32 --height 18 --spp 64 --max-depth 1 --seed 123
                --bias pairwise
    )
    add_test(NAME cuda_restir_spatial_basic_check
        COMMAND cuda_restir_check --mode spatial
                --scene-file assets/scenes/scene_065_restir_many_light.json
                --width 32 --height 18 --spp 1 --max-depth 4 --seed 123
                --bias basic
    )
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
        cuda_restir_gbuffer_check
        cuda_restir_quad_check
        cuda_restir_initial_gi_check
        cuda_restir_gi_reuse_check
        cuda_restir_gi_reuse_basic_check
        cuda_restir_gi_replay_check
        cuda_restir_glossy_gi_initial_check
        cuda_restir_gi_final_gather_check
        cuda_restir_initial_gi_statistics
        cuda_restir_basic_gi_statistics
        cuda_restir_pairwise_gi_statistics
        cuda_restir_glossy_gi_statistics
        cuda_restir_environment_check
        cuda_restir_sphere_mesh_check
        cuda_restir_delta_only_check
        cuda_restir_initial_di_statistics
        cuda_restir_spatial_di_statistics
        cuda_restir_spatiotemporal_di_statistics
        cuda_restir_spatial_basic_check
        cuda_restir_spatial_pairwise_check
        cuda_restir_temporal_pairwise_check
        PROPERTIES
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            SKIP_RETURN_CODE 77
    )
endif()

raytracer_enable_warnings(raytracer_tests)

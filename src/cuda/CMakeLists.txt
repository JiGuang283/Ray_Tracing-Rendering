add_library(raytracer_cuda STATIC
    cuda_renderer.cpp
    device_scene.cu
    intersection_kernels.cu
    light_kernels.cu
    restir/restir_gbuffer.cu
    restir/restir_initial_di.cu
    restir/restir_initial_gi.cu
    restir/restir_spatial_di.cu
    restir/restir_spatial_gi.cu
    restir/restir_temporal_di.cu
    restir/restir_temporal_gi.cu
    restir/restir_reference_shading.cu
    restir/restir_fused_stages.cu
    restir/restir_scheduler.cpp
    restir/restir_workspace.cpp
    shading_kernels.cu
    wavefront_renderer.cu
    workspace_memory.cu
)
add_library(raytracer::cuda ALIAS raytracer_cuda)

target_include_directories(raytracer_cuda PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(raytracer_cuda PUBLIC
    raytracer::build_config
    raytracer::render_data
    raytracer::render_api
    raytracer::restir
    CUDA::cudart
)
set_target_properties(raytracer_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
)
target_compile_options(raytracer_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

raytracer_enable_warnings(raytracer_cuda)

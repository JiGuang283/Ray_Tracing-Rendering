cmake_minimum_required(VERSION 3.20)

project(CGAssignment4 LANGUAGES C CXX)

include(cmake/RaytracerOptions.cmake)
include(cmake/RaytracerDependencies.cmake)
include(CTest)

set(RAYTRACER_HAS_CUDA ${RAYTRACER_ENABLE_CUDA})
set(RAYTRACER_RENDER_DEBUG_ENABLED
    ${RAYTRACER_ENABLE_RENDER_DEBUG}
)
option(RAYTRACER_ENABLE_WARNINGS
       "Compile project targets with -Wall -Wextra -Wpedantic" ON)
add_library(raytracer_warnings INTERFACE)
add_library(raytracer::warnings ALIAS raytracer_warnings)
if(RAYTRACER_ENABLE_WARNINGS)
    if(MSVC)
        target_compile_options(raytracer_warnings INTERFACE /W4)
    else()
        target_compile_options(raytracer_warnings INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall;-Xcompiler=-Wextra>
        )
    endif()
endif()

function(raytracer_enable_warnings target)
    if(TARGET raytracer::warnings)
        target_link_libraries(${target} PRIVATE raytracer::warnings)
    endif()
endfunction()

configure_file(
    cmake/raytracer_build_config.h.in
    ${PROJECT_BINARY_DIR}/generated/raytracer/build_config.h
)

add_library(raytracer_build_config INTERFACE)
add_library(raytracer::build_config ALIAS raytracer_build_config)
target_compile_features(raytracer_build_config INTERFACE cxx_std_17)
target_include_directories(raytracer_build_config INTERFACE
    ${PROJECT_BINARY_DIR}/generated
)

add_subdirectory(src)

if(RAYTRACER_BUILD_TOOLS OR BUILD_TESTING)
    add_subdirectory(tools)
endif()
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

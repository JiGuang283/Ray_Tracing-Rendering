cmake_minimum_required(VERSION 3.20)

project(CGAssignment4 LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_EXTENSIONS OFF)

option(RAYTRACER_ENABLE_CUDA "Build the optional CUDA backend" OFF)

if(RAYTRACER_ENABLE_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING
            "CUDA architectures to compile"
        )
    endif()
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR
            "RAYTRACER_ENABLE_CUDA is ON, but no CUDA compiler was found"
        )
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

find_package(SDL2 REQUIRED)
find_package(Threads REQUIRED)

add_library(raytracer_build_config INTERFACE)
add_library(raytracer::build_config ALIAS raytracer_build_config)
target_compile_features(raytracer_build_config INTERFACE cxx_std_17)

include(CTest)

add_subdirectory(src)
add_subdirectory(tools)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

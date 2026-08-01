cmake_minimum_required(VERSION 3.16)

project(CGAssignment4 LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(RAYTRACER_ENABLE_CUDA "Build the optional CUDA backend" OFF)

if(RAYTRACER_ENABLE_CUDA)
    if(CMAKE_VERSION VERSION_LESS 3.18)
        message(FATAL_ERROR
            "RAYTRACER_ENABLE_CUDA requires CMake 3.18 or newer"
        )
    endif()
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

add_library(mikktspace STATIC
    ${PROJECT_SOURCE_DIR}/src/external/mikktspace/mikktspace.c
)
target_include_directories(mikktspace PUBLIC
    ${PROJECT_SOURCE_DIR}/src/external/mikktspace
)

set(PROJECT_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/app
    ${PROJECT_SOURCE_DIR}/src/core
    ${PROJECT_SOURCE_DIR}/src/external
    ${PROJECT_SOURCE_DIR}/src/geometry
    ${PROJECT_SOURCE_DIR}/src/lighting
    ${PROJECT_SOURCE_DIR}/src/materials
    ${PROJECT_SOURCE_DIR}/src/renderer
    ${PROJECT_SOURCE_DIR}/src/scene
    ${PROJECT_SOURCE_DIR}/src/shading
)

file(GLOB_RECURSE CORE_HEADERS CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/src/*.h
)
file(GLOB_RECURSE CORE_SOURCES CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)
file(GLOB APP_SOURCES CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/src/app/*.cpp
)

list(REMOVE_ITEM CORE_SOURCES
    ${PROJECT_SOURCE_DIR}/src/main.cpp
    ${APP_SOURCES}
)

add_library(raytracer_core STATIC ${CORE_SOURCES} ${CORE_HEADERS})
target_include_directories(raytracer_core PUBLIC ${PROJECT_INCLUDE_DIRS})
target_link_libraries(raytracer_core
    PUBLIC Threads::Threads
    PRIVATE mikktspace
)

if(RAYTRACER_ENABLE_CUDA)
    add_library(raytracer_cuda STATIC
        ${PROJECT_SOURCE_DIR}/src/cuda/device_scene.cu
        ${PROJECT_SOURCE_DIR}/src/cuda/intersection_kernels.cu
    )
    target_include_directories(raytracer_cuda PUBLIC ${PROJECT_INCLUDE_DIRS})
    target_link_libraries(raytracer_cuda
        PUBLIC raytracer_core CUDA::cudart
    )
    set_target_properties(raytracer_cuda PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
    )
    target_compile_options(raytracer_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
endif()

add_executable(${PROJECT_NAME}
    ${PROJECT_SOURCE_DIR}/src/main.cpp
    ${APP_SOURCES}
)
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        raytracer_core
        SDL2
        SDL2main
)

add_executable(scene_data_check
    ${PROJECT_SOURCE_DIR}/tools/scene_data_check.cpp
)
target_include_directories(scene_data_check PRIVATE ${PROJECT_INCLUDE_DIRS})
target_link_libraries(scene_data_check PRIVATE raytracer_core)

add_executable(scene_intersection_check
    ${PROJECT_SOURCE_DIR}/tools/scene_intersection_check.cpp
)
target_include_directories(scene_intersection_check PRIVATE
    ${PROJECT_INCLUDE_DIRS}
)
target_link_libraries(scene_intersection_check PRIVATE raytracer_core)

if(RAYTRACER_ENABLE_CUDA)
    add_executable(cuda_scene_check
        ${PROJECT_SOURCE_DIR}/tools/cuda_scene_check.cpp
    )
    target_include_directories(cuda_scene_check PRIVATE
        ${PROJECT_INCLUDE_DIRS}
    )
    target_link_libraries(cuda_scene_check PRIVATE raytracer_cuda)
endif()

include(CTest)
if(BUILD_TESTING)
    file(GLOB TEST_SOURCES CONFIGURE_DEPENDS
        ${PROJECT_SOURCE_DIR}/tests/*.cpp
    )
    add_executable(raytracer_tests ${TEST_SOURCES})
    target_include_directories(raytracer_tests PRIVATE
        ${PROJECT_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/tests
    )
    target_link_libraries(raytracer_tests PRIVATE raytracer_core)
    add_test(NAME raytracer_tests COMMAND raytracer_tests)
    if(RAYTRACER_ENABLE_CUDA)
        add_test(
            NAME cuda_scene_check
            COMMAND cuda_scene_check --ids 1,7,23 --grid-width 9
                    --grid-height 7
        )
        set_tests_properties(cuda_scene_check PROPERTIES
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            SKIP_RETURN_CODE 77
        )
    endif()
endif()

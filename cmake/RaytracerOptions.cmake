if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_EXTENSIONS OFF)

option(RAYTRACER_BUILD_APP "Build the SDL renderer application" ON)
option(RAYTRACER_BUILD_TOOLS "Build diagnostic command-line tools" ON)
option(RAYTRACER_ENABLE_CUDA "Build the optional CUDA backend" OFF)
option(RAYTRACER_ENABLE_RENDER_DEBUG
       "Compile renderer diagnostics support" OFF)
option(RAYTRACER_ENABLE_LTO
       "Enable link-time optimization for release builds" OFF)
option(RAYTRACER_ENABLE_NATIVE
       "Compile with -march=native (CPU-only builds)" OFF)
if(RAYTRACER_ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()
if(RAYTRACER_ENABLE_NATIVE)
    if(RAYTRACER_ENABLE_CUDA)
        message(WARNING
            "RAYTRACER_ENABLE_NATIVE is intended for CPU-only builds; "
            "not applying -march=native to CUDA compiler")
    else()
        add_compile_options(-march=native)
    endif()
endif()

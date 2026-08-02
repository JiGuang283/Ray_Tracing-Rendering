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

if(RAYTRACER_BUILD_APP)
    find_package(SDL2 REQUIRED)
endif()

find_package(Threads REQUIRED)

#ifndef NORMAL_MAP_TYPES_H
#define NORMAL_MAP_TYPES_H

#include <cstdint>

enum class NormalMapConvention : std::uint32_t {
    OpenGL,
    DirectX
};

struct NormalMapSettings {
    NormalMapConvention convention = NormalMapConvention::OpenGL;
    double strength = 1.0;
};

#endif

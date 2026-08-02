#ifndef IMAGE_SAMPLING_H
#define IMAGE_SAMPLING_H

#include <cstdint>

enum class WrapMode : std::uint32_t {
    Repeat,
    Clamp,
    Mirror
};

enum class FilterMode : std::uint32_t {
    Nearest,
    Bilinear
};

enum class TextureChannel : std::uint32_t {
    RGB,
    R,
    G,
    B,
    A
};

struct SamplerState {
    WrapMode wrap_u = WrapMode::Repeat;
    WrapMode wrap_v = WrapMode::Repeat;
    FilterMode filter = FilterMode::Bilinear;
    bool flip_v = true;
};

#endif

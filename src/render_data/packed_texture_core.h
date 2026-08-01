#ifndef PACKED_TEXTURE_CORE_H
#define PACKED_TEXTURE_CORE_H

#include "packed_texture.h"

#include <cmath>
#include <cstdint>

namespace packed_texture {

struct TextureFrame {
    std::uint32_t texture_id = kInvalidPackedIndex;
    std::uint32_t stage = 0;
    Float2 uv{};
};

static_assert(sizeof(TextureFrame) == 16);
static_assert(std::is_trivially_copyable_v<TextureFrame>);

RT_HOST_DEVICE RT_FORCE_INLINE float minimum(float a, float b) {
    return a < b ? a : b;
}

RT_HOST_DEVICE RT_FORCE_INLINE float maximum(float a, float b) {
    return a > b ? a : b;
}

RT_HOST_DEVICE RT_FORCE_INLINE float clamp(float value, float lower,
                                           float upper) {
    return maximum(lower, minimum(value, upper));
}

RT_HOST_DEVICE RT_FORCE_INLINE Float4 lerp(Float4 a, Float4 b, float t) {
    return {::fmaf(t, b.x - a.x, a.x),
            ::fmaf(t, b.y - a.y, a.y),
            ::fmaf(t, b.z - a.z, a.z),
            ::fmaf(t, b.w - a.w, a.w)};
}

RT_HOST_DEVICE RT_FORCE_INLINE Float4 multiply(Float4 a, Float4 b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};
}

RT_HOST_DEVICE RT_FORCE_INLINE float srgb_to_linear(float value) {
    value = clamp(value, 0.0f, 1.0f);
    if (value <= 0.04045f) {
        return value / 12.92f;
    }
    return ::powf((value + 0.055f) / 1.055f, 2.4f);
}

RT_HOST_DEVICE RT_FORCE_INLINE std::uint32_t sampler_field(
    std::uint32_t flags, std::uint32_t shift) {
    return (flags >> shift) & 0x3u;
}

RT_HOST_DEVICE RT_FORCE_INLINE float wrap_coordinate(float value,
                                                      std::uint32_t mode) {
    if (mode == 1u) {
        return clamp(value, 0.0f, 1.0f);
    }
    if (mode == 0u) {
        return value - ::floorf(value);
    }
    float mirrored = ::fmodf(value, 2.0f);
    if (mirrored < 0.0f) {
        mirrored += 2.0f;
    }
    return mirrored <= 1.0f ? mirrored : 2.0f - mirrored;
}

RT_HOST_DEVICE RT_FORCE_INLINE int wrap_index(int index, int size,
                                               std::uint32_t mode) {
    if (mode == 1u) {
        const int upper = size - 1;
        return index < 0 ? 0 : (index > upper ? upper : index);
    }
    if (mode == 0u) {
        const int wrapped = index % size;
        return wrapped < 0 ? wrapped + size : wrapped;
    }
    const int period = 2 * size;
    int wrapped = index % period;
    if (wrapped < 0) {
        wrapped += period;
    }
    return wrapped < size ? wrapped : period - wrapped - 1;
}

RT_HOST_DEVICE RT_FORCE_INLINE float image_component(
    const CompiledSceneView &scene, const PackedImageDesc &image, int x,
    int y, std::uint32_t channel) {
    const int max_x = static_cast<int>(image.width) - 1;
    const int max_y = static_cast<int>(image.height) - 1;
    x = x < 0 ? 0 : (x > max_x ? max_x : x);
    y = y < 0 ? 0 : (y > max_y ? max_y : y);
    std::uint32_t source_channel = channel;
    if (image.channels == 1) {
        if (channel == 3) {
            return 1.0f;
        }
        source_channel = 0;
    } else if (image.channels == 2) {
        source_channel = channel == 3 ? 1u : 0u;
    } else if (channel >= image.channels) {
        if (channel == 3) {
            return 1.0f;
        }
        source_channel = 0;
    }
    const std::uint64_t texel =
        (static_cast<std::uint64_t>(y) * image.width +
         static_cast<std::uint32_t>(x)) *
            image.channels +
        source_channel;
    return scene.image_texels[
        image.texels.offset + static_cast<std::uint32_t>(texel)];
}

RT_HOST_DEVICE RT_FORCE_INLINE Float4 image_texel(
    const CompiledSceneView &scene, const PackedTextureNode &texture,
    const PackedImageDesc &image, int x, int y) {
    Float4 value{image_component(scene, image, x, y, 0),
                 image_component(scene, image, x, y, 1),
                 image_component(scene, image, x, y, 2),
                 image_component(scene, image, x, y, 3)};
    if ((texture.sampler_flags & PACKED_SAMPLER_SRGB) != 0 &&
        (image.flags & PACKED_IMAGE_HDR) == 0) {
        value.x = srgb_to_linear(value.x);
        value.y = srgb_to_linear(value.y);
        value.z = srgb_to_linear(value.z);
    }
    if (texture.channel != 0u) {
        float channel = value.x;
        switch (texture.channel) {
        case 1:
            channel = value.x;
            break;
        case 2:
            channel = value.y;
            break;
        case 3:
            channel = value.z;
            break;
        case 4:
            channel = value.w;
            break;
        default:
            break;
        }
        value.x = channel;
        value.y = channel;
        value.z = channel;
    }
    return value;
}

RT_HOST_DEVICE RT_FORCE_INLINE Float4 evaluate_image(
    const CompiledSceneView &scene, const PackedTextureNode &texture,
    Float2 uv) {
    const PackedImageDesc &image = scene.images[texture.image_id];
    const std::uint32_t wrap_u = sampler_field(
        texture.sampler_flags, PACKED_SAMPLER_WRAP_U_SHIFT);
    const std::uint32_t wrap_v = sampler_field(
        texture.sampler_flags, PACKED_SAMPLER_WRAP_V_SHIFT);
    const float source_v =
        (texture.sampler_flags & PACKED_SAMPLER_FLIP_V) != 0
            ? 1.0f - uv.y
            : uv.y;
    const float u = wrap_coordinate(uv.x, wrap_u);
    const float v = wrap_coordinate(source_v, wrap_v);
    const bool bilinear =
        ((texture.sampler_flags >> PACKED_SAMPLER_FILTER_SHIFT) & 0x1u) !=
        0;
    if (!bilinear) {
        const int x = wrap_index(
            static_cast<int>(::floorf(u * image.width)),
            static_cast<int>(image.width), wrap_u);
        const int y = wrap_index(
            static_cast<int>(::floorf(v * image.height)),
            static_cast<int>(image.height), wrap_v);
        return image_texel(scene, texture, image, x, y);
    }

    const float image_x = u * image.width - 0.5f;
    const float image_y = v * image.height - 0.5f;
    const int x0 = static_cast<int>(::floorf(image_x));
    const int y0 = static_cast<int>(::floorf(image_y));
    const float tx = image_x - x0;
    const float ty = image_y - y0;
    const Float4 s00 = image_texel(
        scene, texture, image,
        wrap_index(x0, static_cast<int>(image.width), wrap_u),
        wrap_index(y0, static_cast<int>(image.height), wrap_v));
    const Float4 s10 = image_texel(
        scene, texture, image,
        wrap_index(x0 + 1, static_cast<int>(image.width), wrap_u),
        wrap_index(y0, static_cast<int>(image.height), wrap_v));
    const Float4 s01 = image_texel(
        scene, texture, image,
        wrap_index(x0, static_cast<int>(image.width), wrap_u),
        wrap_index(y0 + 1, static_cast<int>(image.height), wrap_v));
    const Float4 s11 = image_texel(
        scene, texture, image,
        wrap_index(x0 + 1, static_cast<int>(image.width), wrap_u),
        wrap_index(y0 + 1, static_cast<int>(image.height), wrap_v));
    return lerp(lerp(s00, s10, tx), lerp(s01, s11, tx), ty);
}

RT_HOST_DEVICE RT_FORCE_INLINE float perlin_noise(
    const CompiledSceneView &scene, const PackedPerlinDesc &table,
    Float3 point) {
    const float u = point.x - ::floorf(point.x);
    const float v = point.y - ::floorf(point.y);
    const float w = point.z - ::floorf(point.z);
    const int base_x = static_cast<int>(::floorf(point.x));
    const int base_y = static_cast<int>(::floorf(point.y));
    const int base_z = static_cast<int>(::floorf(point.z));
    const float uu = u * u * (3.0f - 2.0f * u);
    const float vv = v * v * (3.0f - 2.0f * v);
    const float ww = w * w * (3.0f - 2.0f * w);
    float result = 0.0f;
    for (int x = 0; x < 2; ++x) {
        for (int y = 0; y < 2; ++y) {
            for (int z = 0; z < 2; ++z) {
                const std::uint32_t px = scene.perlin_permutations[
                    table.permutation_x.offset + ((base_x + x) & 255)];
                const std::uint32_t py = scene.perlin_permutations[
                    table.permutation_y.offset + ((base_y + y) & 255)];
                const std::uint32_t pz = scene.perlin_permutations[
                    table.permutation_z.offset + ((base_z + z) & 255)];
                const Float4 gradient = scene.perlin_gradients[
                    table.gradients.offset + (px ^ py ^ pz)];
                const float gradient_dot =
                    ::fmaf(gradient.x, u - x,
                           ::fmaf(gradient.y, v - y,
                                  gradient.z * (w - z)));
                result += (x * uu + (1 - x) * (1.0f - uu)) *
                          (y * vv + (1 - y) * (1.0f - vv)) *
                          (z * ww + (1 - z) * (1.0f - ww)) * gradient_dot;
            }
        }
    }
    return result;
}

RT_HOST_DEVICE RT_FORCE_INLINE float perlin_turbulence(
    const CompiledSceneView &scene, const PackedPerlinDesc &table,
    Float3 point) {
    float result = 0.0f;
    float weight = 1.0f;
    for (int depth = 0; depth < 7; ++depth) {
        result += weight * perlin_noise(scene, table, point);
        weight *= 0.5f;
        point.x *= 2.0f;
        point.y *= 2.0f;
        point.z *= 2.0f;
    }
    return result < 0.0f ? -result : result;
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedShadingStatus push_frame(
    TextureFrame *frames, std::uint32_t &frame_count,
    std::uint32_t texture_id, Float2 uv, std::uint32_t &max_stack_used) {
    if (texture_id == kInvalidPackedIndex) {
        return PackedShadingStatus::TextureFailure;
    }
    if (frame_count >= kPackedTextureStackCapacity) {
        return PackedShadingStatus::TextureStackOverflow;
    }
    frames[frame_count++] = {texture_id, 0, uv};
    if (frame_count > max_stack_used) {
        max_stack_used = frame_count;
    }
    return PackedShadingStatus::Success;
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedShadingStatus push_value(
    Float4 *values, std::uint32_t &value_count, Float4 value) {
    if (value_count >= kPackedTextureValueCapacity) {
        return PackedShadingStatus::TextureStackOverflow;
    }
    values[value_count++] = value;
    return PackedShadingStatus::Success;
}

RT_HOST_DEVICE RT_FORCE_INLINE PackedShadingStatus
evaluate_packed_texture_core(const CompiledSceneView &scene,
                             std::uint32_t texture_id,
                             const PackedTextureEvalContext &context,
                             Float4 &sample,
                             std::uint32_t *reported_max_stack = nullptr) {
    if (texture_id >= scene.texture_nodes.count) {
        return PackedShadingStatus::TextureFailure;
    }
    TextureFrame frames[kPackedTextureStackCapacity]{};
    Float4 values[kPackedTextureValueCapacity]{};
    std::uint32_t frame_count = 0;
    std::uint32_t value_count = 0;
    std::uint32_t max_stack_used = 0;
    PackedShadingStatus status = push_frame(
        frames, frame_count, texture_id, context.uv0, max_stack_used);
    if (status != PackedShadingStatus::Success) {
        return status;
    }

    while (frame_count != 0) {
        TextureFrame &frame = frames[frame_count - 1];
        if (frame.texture_id >= scene.texture_nodes.count) {
            return PackedShadingStatus::TextureFailure;
        }
        const PackedTextureNode &node =
            scene.texture_nodes[frame.texture_id];
        Float4 result{};
        switch (node.type) {
        case PackedTextureType::Constant:
            result = node.value0;
            --frame_count;
            status = push_value(values, value_count, result);
            break;
        case PackedTextureType::VertexColor:
            result = context.vertex_color;
            --frame_count;
            status = push_value(values, value_count, result);
            break;
        case PackedTextureType::Checker:
            if (frame.stage == 0) {
                frame.stage = 1;
                const float pattern =
                    ::sinf(10.0f * context.position.x) *
                    ::sinf(10.0f * context.position.y) *
                    ::sinf(10.0f * context.position.z);
                status = push_frame(
                    frames, frame_count,
                    pattern < 0.0f ? node.input1 : node.input0, frame.uv,
                    max_stack_used);
            } else {
                --frame_count;
            }
            break;
        case PackedTextureType::Noise:
            if (node.perlin_id >= scene.perlin_tables.count) {
                return PackedShadingStatus::TextureFailure;
            }
            result.x =
                0.5f *
                (1.0f +
                 ::sinf(node.value0.x * context.position.z +
                        10.0f * perlin_turbulence(
                                    scene,
                                    scene.perlin_tables[node.perlin_id],
                                    context.position)));
            result.y = result.x;
            result.z = result.x;
            result.w = 1.0f;
            --frame_count;
            status = push_value(values, value_count, result);
            break;
        case PackedTextureType::Image:
            if (node.image_id >= scene.images.count) {
                return PackedShadingStatus::TextureFailure;
            }
            result = evaluate_image(scene, node, frame.uv);
            --frame_count;
            status = push_value(values, value_count, result);
            break;
        case PackedTextureType::Scale:
            if (frame.stage == 0) {
                frame.stage = 1;
                status = push_frame(frames, frame_count, node.input0,
                                    frame.uv, max_stack_used);
            } else {
                if (value_count == 0) {
                    return PackedShadingStatus::TextureFailure;
                }
                values[value_count - 1].x *= node.value0.x;
                values[value_count - 1].y *= node.value0.x;
                values[value_count - 1].z *= node.value0.x;
                --frame_count;
            }
            break;
        case PackedTextureType::UVTransform:
            if (frame.stage == 0) {
                frame.stage = 1;
                const float scaled_u = frame.uv.x * node.value0.z;
                const float scaled_v = frame.uv.y * node.value0.w;
                const Float2 transformed{
                    node.value0.x + node.value1.x * scaled_u -
                        node.value1.y * scaled_v,
                    node.value0.y + node.value1.y * scaled_u +
                        node.value1.x * scaled_v};
                status = push_frame(frames, frame_count, node.input0,
                                    transformed, max_stack_used);
            } else {
                --frame_count;
            }
            break;
        case PackedTextureType::Multiply:
            if (frame.stage == 0) {
                frame.stage = 1;
                status = push_frame(frames, frame_count, node.input0,
                                    frame.uv, max_stack_used);
            } else if (frame.stage == 1) {
                frame.stage = 2;
                status = push_frame(frames, frame_count, node.input1,
                                    frame.uv, max_stack_used);
            } else {
                if (value_count < 2) {
                    return PackedShadingStatus::TextureFailure;
                }
                const Float4 b = values[--value_count];
                const Float4 a = values[--value_count];
                --frame_count;
                status = push_value(values, value_count, multiply(a, b));
            }
            break;
        case PackedTextureType::Mix:
            if (frame.stage == 0) {
                frame.stage = 1;
                status = push_frame(frames, frame_count, node.input0,
                                    frame.uv, max_stack_used);
            } else if (frame.stage == 1) {
                frame.stage = 2;
                status = push_frame(frames, frame_count, node.input1,
                                    frame.uv, max_stack_used);
            } else if (frame.stage == 2) {
                frame.stage = 3;
                status = push_frame(frames, frame_count, node.input2,
                                    frame.uv, max_stack_used);
            } else {
                if (value_count < 3) {
                    return PackedShadingStatus::TextureFailure;
                }
                const Float4 factor = values[--value_count];
                const Float4 b = values[--value_count];
                const Float4 a = values[--value_count];
                --frame_count;
                status = push_value(
                    values, value_count,
                    lerp(a, b, clamp(factor.x, 0.0f, 1.0f)));
            }
            break;
        case PackedTextureType::ColorRamp:
            if (frame.stage == 0) {
                frame.stage = 1;
                status = push_frame(frames, frame_count, node.input0,
                                    frame.uv, max_stack_used);
            } else {
                if (value_count == 0) {
                    return PackedShadingStatus::TextureFailure;
                }
                const Float4 value = values[value_count - 1];
                const float denominator = node.value2.y - node.value2.x;
                const float factor =
                    denominator == 0.0f
                        ? 0.0f
                        : clamp((value.x - node.value2.x) / denominator,
                                0.0f, 1.0f);
                values[value_count - 1] =
                    lerp(node.value0, node.value1, factor);
                values[value_count - 1].w = 1.0f;
                --frame_count;
            }
            break;
        }
        if (status != PackedShadingStatus::Success) {
            return status;
        }
    }

    if (value_count != 1) {
        return PackedShadingStatus::TextureFailure;
    }
    sample = values[0];
    if (reported_max_stack != nullptr) {
        *reported_max_stack = max_stack_used;
    }
    return PackedShadingStatus::Success;
}

} // namespace packed_texture

#endif

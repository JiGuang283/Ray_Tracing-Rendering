#include "packed_texture.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {

constexpr std::uint32_t kMaxTextureDepth = 64;

Float4 lerp(Float4 a, Float4 b, float t) {
    return {(1.0f - t) * a.x + t * b.x,
            (1.0f - t) * a.y + t * b.y,
            (1.0f - t) * a.z + t * b.z,
            (1.0f - t) * a.w + t * b.w};
}

Float4 multiply(Float4 a, Float4 b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};
}

float srgb_to_linear(float value) {
    value = std::clamp(value, 0.0f, 1.0f);
    if (value <= 0.04045f) {
        return value / 12.92f;
    }
    return std::pow((value + 0.055f) / 1.055f, 2.4f);
}

std::uint32_t sampler_field(std::uint32_t flags, std::uint32_t shift) {
    return (flags >> shift) & 0x3u;
}

float wrap_coordinate(float value, std::uint32_t mode) {
    if (mode == 1u) {
        return std::clamp(value, 0.0f, 1.0f);
    }
    if (mode == 0u) {
        return value - std::floor(value);
    }
    float mirrored = std::fmod(value, 2.0f);
    if (mirrored < 0.0f) {
        mirrored += 2.0f;
    }
    return mirrored <= 1.0f ? mirrored : 2.0f - mirrored;
}

int wrap_index(int index, int size, std::uint32_t mode) {
    if (mode == 1u) {
        return std::max(0, std::min(index, size - 1));
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

float image_component(const CompiledSceneView &scene,
                      const PackedImageDesc &image, int x, int y,
                      std::uint32_t channel) {
    x = std::max(0, std::min(x, static_cast<int>(image.width) - 1));
    y = std::max(0, std::min(y, static_cast<int>(image.height) - 1));
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

Float4 image_texel(const CompiledSceneView &scene,
                   const PackedTextureNode &texture,
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

Float4 evaluate_image(const CompiledSceneView &scene,
                      const PackedTextureNode &texture,
                      const PackedTextureEvalContext &context) {
    const PackedImageDesc &image = scene.images[texture.image_id];
    const std::uint32_t wrap_u = sampler_field(
        texture.sampler_flags, PACKED_SAMPLER_WRAP_U_SHIFT);
    const std::uint32_t wrap_v = sampler_field(
        texture.sampler_flags, PACKED_SAMPLER_WRAP_V_SHIFT);
    const float source_v =
        (texture.sampler_flags & PACKED_SAMPLER_FLIP_V) != 0
            ? 1.0f - context.uv0.y
            : context.uv0.y;
    const float u = wrap_coordinate(context.uv0.x, wrap_u);
    const float v = wrap_coordinate(source_v, wrap_v);
    const bool bilinear =
        ((texture.sampler_flags >> PACKED_SAMPLER_FILTER_SHIFT) & 0x1u) !=
        0;
    if (!bilinear) {
        const int x = wrap_index(
            static_cast<int>(std::floor(u * image.width)),
            static_cast<int>(image.width), wrap_u);
        const int y = wrap_index(
            static_cast<int>(std::floor(v * image.height)),
            static_cast<int>(image.height), wrap_v);
        return image_texel(scene, texture, image, x, y);
    }

    const float image_x = u * image.width - 0.5f;
    const float image_y = v * image.height - 0.5f;
    const int x0 = static_cast<int>(std::floor(image_x));
    const int y0 = static_cast<int>(std::floor(image_y));
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

float perlin_noise(const CompiledSceneView &scene,
                   const PackedPerlinDesc &table, Float3 point) {
    const float u = point.x - std::floor(point.x);
    const float v = point.y - std::floor(point.y);
    const float w = point.z - std::floor(point.z);
    const int base_x = static_cast<int>(std::floor(point.x));
    const int base_y = static_cast<int>(std::floor(point.y));
    const int base_z = static_cast<int>(std::floor(point.z));
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
                    gradient.x * (u - x) + gradient.y * (v - y) +
                    gradient.z * (w - z);
                result += (x * uu + (1 - x) * (1.0f - uu)) *
                          (y * vv + (1 - y) * (1.0f - vv)) *
                          (z * ww + (1 - z) * (1.0f - ww)) * gradient_dot;
            }
        }
    }
    return result;
}

float perlin_turbulence(const CompiledSceneView &scene,
                        const PackedPerlinDesc &table, Float3 point) {
    float result = 0.0f;
    float weight = 1.0f;
    for (int depth = 0; depth < 7; ++depth) {
        result += weight * perlin_noise(scene, table, point);
        weight *= 0.5f;
        point.x *= 2.0f;
        point.y *= 2.0f;
        point.z *= 2.0f;
    }
    return std::abs(result);
}

bool evaluate_node(const CompiledSceneView &scene, std::uint32_t texture_id,
                   const PackedTextureEvalContext &context, Float4 &sample,
                   std::uint32_t depth) {
    if (texture_id >= scene.texture_nodes.count ||
        depth >= kMaxTextureDepth) {
        return false;
    }
    const PackedTextureNode &node = scene.texture_nodes[texture_id];
    auto input = [&](std::uint32_t id, const PackedTextureEvalContext &ctx,
                     Float4 &value) {
        return evaluate_node(scene, id, ctx, value, depth + 1);
    };
    switch (node.type) {
    case PackedTextureType::Constant:
        sample = node.value0;
        return true;
    case PackedTextureType::VertexColor:
        sample = context.vertex_color;
        return true;
    case PackedTextureType::Checker: {
        const float pattern = std::sin(10.0f * context.position.x) *
                              std::sin(10.0f * context.position.y) *
                              std::sin(10.0f * context.position.z);
        return input(pattern < 0.0f ? node.input1 : node.input0, context,
                     sample);
    }
    case PackedTextureType::Noise: {
        if (node.perlin_id >= scene.perlin_tables.count) {
            return false;
        }
        const float value =
            0.5f *
            (1.0f +
             std::sin(node.value0.x * context.position.z +
                      10.0f * perlin_turbulence(
                                  scene, scene.perlin_tables[node.perlin_id],
                                  context.position)));
        sample = {value, value, value, 1.0f};
        return true;
    }
    case PackedTextureType::Image:
        if (node.image_id >= scene.images.count) {
            return false;
        }
        sample = evaluate_image(scene, node, context);
        return true;
    case PackedTextureType::Scale:
        if (!input(node.input0, context, sample)) {
            return false;
        }
        sample.x *= node.value0.x;
        sample.y *= node.value0.x;
        sample.z *= node.value0.x;
        return true;
    case PackedTextureType::UVTransform: {
        PackedTextureEvalContext transformed = context;
        const float scaled_u = context.uv0.x * node.value0.z;
        const float scaled_v = context.uv0.y * node.value0.w;
        transformed.uv0 =
            {node.value0.x + node.value1.x * scaled_u -
                                 node.value1.y * scaled_v,
             node.value0.y + node.value1.y * scaled_u +
                                 node.value1.x * scaled_v};
        return input(node.input0, transformed, sample);
    }
    case PackedTextureType::Multiply: {
        Float4 a;
        Float4 b;
        if (!input(node.input0, context, a) ||
            !input(node.input1, context, b)) {
            return false;
        }
        sample = multiply(a, b);
        return true;
    }
    case PackedTextureType::Mix: {
        Float4 a;
        Float4 b;
        Float4 factor;
        if (!input(node.input0, context, a) ||
            !input(node.input1, context, b) ||
            !input(node.input2, context, factor)) {
            return false;
        }
        sample = lerp(a, b, std::clamp(factor.x, 0.0f, 1.0f));
        return true;
    }
    case PackedTextureType::ColorRamp: {
        Float4 value;
        if (!input(node.input0, context, value)) {
            return false;
        }
        const float denominator = node.value2.y - node.value2.x;
        const float factor =
            denominator == 0.0f
                ? 0.0f
                : std::clamp((value.x - node.value2.x) / denominator,
                             0.0f, 1.0f);
        sample = lerp(node.value0, node.value1, factor);
        sample.w = 1.0f;
        return true;
    }
    }
    return false;
}

} // namespace

bool evaluate_packed_texture(const CompiledSceneView &scene,
                             std::uint32_t texture_id,
                             const PackedTextureEvalContext &context,
                             Float4 &sample) {
    return evaluate_node(scene, texture_id, context, sample, 0);
}

#ifndef TEXTURE_H
#define TEXTURE_H

#include "image_asset.h"
#include "perlin.h"
#include "shader_context.h"

#include <memory>
#include <cstdint>

enum class ColorSpace {
    SRGB,
    Linear
};

enum class WrapMode {
    Repeat,
    Clamp,
    Mirror
};

enum class FilterMode {
    Nearest,
    Bilinear
};

enum class TextureChannel {
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

struct TextureSample {
    color rgb{0, 0, 0};
    double alpha = 1.0;
};

enum class TextureKind : std::uint32_t {
    Unsupported = 0,
    SolidColor,
    VertexColor,
    Checker,
    Image,
    Noise,
    Scale,
    UVTransform,
    Multiply,
    Mix,
    ColorRamp
};

class Texture {
  public:
    virtual ~Texture() = default;
    virtual TextureSample
    evaluate(const ShaderEvalContext &context) const = 0;
    virtual TextureKind kind() const {
        return TextureKind::Unsupported;
    }
};

using TextureHandle = std::shared_ptr<const Texture>;

double srgb_to_linear(double value);
double texture_scalar(const TextureHandle &texture,
                      const ShaderEvalContext &context);

class SolidColorTexture final : public Texture {
  public:
    explicit SolidColorTexture(const color &value);
    SolidColorTexture(double red, double green, double blue);

    TextureSample
    evaluate(const ShaderEvalContext &context) const override;
    TextureKind kind() const override;
    const color &value() const;

  private:
    color m_value;
};

class VertexColorTexture final : public Texture {
  public:
    TextureSample
    evaluate(const ShaderEvalContext &context) const override;
    TextureKind kind() const override;
};

class CheckerTexture final : public Texture {
  public:
    CheckerTexture(TextureHandle even, TextureHandle odd);
    CheckerTexture(const color &even, const color &odd);

    TextureSample
    evaluate(const ShaderEvalContext &context) const override;
    TextureKind kind() const override;
    const TextureHandle &even() const;
    const TextureHandle &odd() const;

  private:
    TextureHandle m_even;
    TextureHandle m_odd;
};

class ImageTexture final : public Texture {
  public:
    ImageTexture(std::shared_ptr<const ImageAsset> image,
                 ColorSpace color_space = ColorSpace::SRGB,
                 SamplerState sampler = {},
                 TextureChannel channel = TextureChannel::RGB);

    TextureSample
    evaluate(const ShaderEvalContext &context) const override;
    TextureKind kind() const override;
    const std::shared_ptr<const ImageAsset> &image() const;
    ColorSpace color_space() const;
    const SamplerState &sampler() const;
    TextureChannel channel() const;

  private:
    TextureSample texel(int x, int y) const;
    int wrap_index(int index, int size, WrapMode mode) const;
    double wrap_coordinate(double value, WrapMode mode) const;

    std::shared_ptr<const ImageAsset> m_image;
    ColorSpace m_color_space;
    SamplerState m_sampler;
    TextureChannel m_channel;
};

class NoiseTexture final : public Texture {
  public:
    explicit NoiseTexture(double scale = 1.0);

    TextureSample
    evaluate(const ShaderEvalContext &context) const override;
    TextureKind kind() const override;
    double scale() const;
    const perlin &noise_data() const;

  private:
    perlin m_noise;
    double m_scale;
};

class ScaleTexture final : public Texture {
  public:
    ScaleTexture(TextureHandle input, double scale);

    TextureSample
    evaluate(const ShaderEvalContext &context) const override;
    TextureKind kind() const override;
    const TextureHandle &input() const;
    double scale() const;

  private:
    TextureHandle m_input;
    double m_scale;
};

class UVTransformTexture final : public Texture {
  public:
    UVTransformTexture(TextureHandle input, const vec2 &offset,
                       const vec2 &scale, double rotation_radians);

    TextureSample
    evaluate(const ShaderEvalContext &context) const override;
    TextureKind kind() const override;
    const TextureHandle &input() const;
    const vec2 &offset() const;
    const vec2 &scale() const;
    double cos_rotation() const;
    double sin_rotation() const;

  private:
    TextureHandle m_input;
    vec2 m_offset;
    vec2 m_scale;
    double m_cos_rotation = 1.0;
    double m_sin_rotation = 0.0;
};

class MultiplyTexture final : public Texture {
  public:
    MultiplyTexture(TextureHandle a, TextureHandle b);

    TextureSample
    evaluate(const ShaderEvalContext &context) const override;
    TextureKind kind() const override;
    const TextureHandle &a() const;
    const TextureHandle &b() const;

  private:
    TextureHandle m_a;
    TextureHandle m_b;
};

class MixTexture final : public Texture {
  public:
    MixTexture(TextureHandle a, TextureHandle b, TextureHandle factor);

    TextureSample
    evaluate(const ShaderEvalContext &context) const override;
    TextureKind kind() const override;
    const TextureHandle &a() const;
    const TextureHandle &b() const;
    const TextureHandle &factor() const;

  private:
    TextureHandle m_a;
    TextureHandle m_b;
    TextureHandle m_factor;
};

class ColorRampTexture final : public Texture {
  public:
    ColorRampTexture(TextureHandle input, const color &low,
                     const color &high, double min_value = 0.0,
                     double max_value = 1.0);

    TextureSample
    evaluate(const ShaderEvalContext &context) const override;
    TextureKind kind() const override;
    const TextureHandle &input() const;
    const color &low() const;
    const color &high() const;
    double min_value() const;
    double max_value() const;

  private:
    TextureHandle m_input;
    color m_low;
    color m_high;
    double m_min_value;
    double m_max_value;
};

#endif

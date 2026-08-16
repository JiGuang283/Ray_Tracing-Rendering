#include "texture.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace {

color lerp(const color &a, const color &b, double t) {
    return (1.0 - t) * a + t * b;
}

double channel_value(const color &rgb, double alpha,
                     TextureChannel channel) {
    switch (channel) {
    case TextureChannel::R:
        return rgb.x();
    case TextureChannel::G:
        return rgb.y();
    case TextureChannel::B:
        return rgb.z();
    case TextureChannel::A:
        return alpha;
    case TextureChannel::RGB:
        return rgb.x();
    }
    return rgb.x();
}

} // namespace

double srgb_to_linear(double value) {
    value = clamp(value, 0.0, 1.0);
    if (value <= 0.04045) {
        return value / 12.92;
    }
    return std::pow((value + 0.055) / 1.055, 2.4);
}

double texture_scalar(const TextureHandle &texture,
                      const ShaderEvalContext &context) {
    return texture ? texture->evaluate(context).rgb.x() : 0.0;
}

SolidColorTexture::SolidColorTexture(const color &value) : m_value(value) {
}

SolidColorTexture::SolidColorTexture(double red, double green, double blue)
    : m_value(red, green, blue) {
}

TextureSample
SolidColorTexture::evaluate(const ShaderEvalContext & /*context*/) const {
    return TextureSample{m_value, 1.0};
}

TextureKind SolidColorTexture::kind() const {
    return TextureKind::SolidColor;
}

const color &SolidColorTexture::value() const {
    return m_value;
}

TextureSample
VertexColorTexture::evaluate(const ShaderEvalContext &context) const {
    return TextureSample{context.vertex_color, context.vertex_alpha};
}

TextureKind VertexColorTexture::kind() const {
    return TextureKind::VertexColor;
}

CheckerTexture::CheckerTexture(TextureHandle even, TextureHandle odd)
    : m_even(std::move(even)), m_odd(std::move(odd)) {
}

CheckerTexture::CheckerTexture(const color &even, const color &odd)
    : m_even(std::make_shared<SolidColorTexture>(even)),
      m_odd(std::make_shared<SolidColorTexture>(odd)) {
}

TextureSample
CheckerTexture::evaluate(const ShaderEvalContext &context) const {
    const point3 &p = context.position;
    const double pattern =
        std::sin(10.0 * p.x()) * std::sin(10.0 * p.y()) *
        std::sin(10.0 * p.z());
    return (pattern < 0.0 ? m_odd : m_even)->evaluate(context);
}

TextureKind CheckerTexture::kind() const {
    return TextureKind::Checker;
}

const TextureHandle &CheckerTexture::even() const {
    return m_even;
}

const TextureHandle &CheckerTexture::odd() const {
    return m_odd;
}

ImageTexture::ImageTexture(std::shared_ptr<const ImageAsset> image,
                           ColorSpace color_space, SamplerState sampler,
                           TextureChannel channel)
    : m_image(image ? std::move(image) : ImageAsset::diagnostic()),
      m_color_space(color_space), m_sampler(sampler), m_channel(channel) {
}

TextureSample ImageTexture::evaluate(const ShaderEvalContext &context) const {
    const int width = m_image->width();
    const int height = m_image->height();
    double u = wrap_coordinate(context.uv0.x(), m_sampler.wrap_u);
    const double source_v =
        m_sampler.flip_v ? 1.0 - context.uv0.y() : context.uv0.y();
    double v = wrap_coordinate(source_v, m_sampler.wrap_v);

    if (m_sampler.filter == FilterMode::Nearest) {
        const int x = wrap_index(static_cast<int>(std::floor(u * width)),
                                 width, m_sampler.wrap_u);
        const int y = wrap_index(static_cast<int>(std::floor(v * height)),
                                 height, m_sampler.wrap_v);
        return texel(x, y);
    }

    const double image_x = u * width - 0.5;
    const double image_y = v * height - 0.5;
    const int x0 = static_cast<int>(std::floor(image_x));
    const int y0 = static_cast<int>(std::floor(image_y));
    const double tx = image_x - x0;
    const double ty = image_y - y0;

    const TextureSample s00 =
        texel(wrap_index(x0, width, m_sampler.wrap_u),
              wrap_index(y0, height, m_sampler.wrap_v));
    const TextureSample s10 =
        texel(wrap_index(x0 + 1, width, m_sampler.wrap_u),
              wrap_index(y0, height, m_sampler.wrap_v));
    const TextureSample s01 =
        texel(wrap_index(x0, width, m_sampler.wrap_u),
              wrap_index(y0 + 1, height, m_sampler.wrap_v));
    const TextureSample s11 =
        texel(wrap_index(x0 + 1, width, m_sampler.wrap_u),
              wrap_index(y0 + 1, height, m_sampler.wrap_v));

    const color row0 = lerp(s00.rgb, s10.rgb, tx);
    const color row1 = lerp(s01.rgb, s11.rgb, tx);
    const double alpha0 = (1.0 - tx) * s00.alpha + tx * s10.alpha;
    const double alpha1 = (1.0 - tx) * s01.alpha + tx * s11.alpha;
    return TextureSample{lerp(row0, row1, ty),
                         (1.0 - ty) * alpha0 + ty * alpha1};
}

const std::shared_ptr<const ImageAsset> &ImageTexture::image() const {
    return m_image;
}

TextureKind ImageTexture::kind() const {
    return TextureKind::Image;
}

ColorSpace ImageTexture::color_space() const {
    return m_color_space;
}

const SamplerState &ImageTexture::sampler() const {
    return m_sampler;
}

TextureChannel ImageTexture::channel() const {
    return m_channel;
}

TextureSample ImageTexture::texel(int x, int y) const {
    color rgb;
    if (m_color_space == ColorSpace::SRGB && !m_image->is_hdr()) {
        rgb = color(m_image->linear_component(x, y, 0),
                    m_image->linear_component(x, y, 1),
                    m_image->linear_component(x, y, 2));
    } else {
        rgb = color(m_image->component(x, y, 0),
                    m_image->component(x, y, 1),
                    m_image->component(x, y, 2));
    }
    const double alpha = m_image->component(x, y, 3);
    if (m_channel != TextureChannel::RGB) {
        const double scalar = channel_value(rgb, alpha, m_channel);
        rgb = color(scalar, scalar, scalar);
    }
    return TextureSample{rgb, alpha};
}

int ImageTexture::wrap_index(int index, int size, WrapMode mode) const {
    if (mode == WrapMode::Clamp) {
        return std::max(0, std::min(index, size - 1));
    }
    if (mode == WrapMode::Repeat) {
        int wrapped = index % size;
        return wrapped < 0 ? wrapped + size : wrapped;
    }

    const int period = 2 * size;
    int wrapped = index % period;
    if (wrapped < 0) {
        wrapped += period;
    }
    return wrapped < size ? wrapped : period - wrapped - 1;
}

double ImageTexture::wrap_coordinate(double value, WrapMode mode) const {
    if (mode == WrapMode::Clamp) {
        return clamp(value, 0.0, 1.0);
    }
    if (mode == WrapMode::Repeat) {
        return value - std::floor(value);
    }

    double mirrored = std::fmod(value, 2.0);
    if (mirrored < 0.0) {
        mirrored += 2.0;
    }
    return mirrored <= 1.0 ? mirrored : 2.0 - mirrored;
}

NoiseTexture::NoiseTexture(double scale) : m_scale(scale) {
}

TextureSample NoiseTexture::evaluate(const ShaderEvalContext &context) const {
    const double value =
        0.5 *
        (1.0 + std::sin(m_scale * context.position.z() +
                        10.0 * m_noise.turb(context.position)));
    return TextureSample{color(value, value, value), 1.0};
}

TextureKind NoiseTexture::kind() const {
    return TextureKind::Noise;
}

double NoiseTexture::scale() const {
    return m_scale;
}

const perlin &NoiseTexture::noise_data() const {
    return m_noise;
}

ScaleTexture::ScaleTexture(TextureHandle input, double scale)
    : m_input(std::move(input)), m_scale(scale) {
}

TextureSample ScaleTexture::evaluate(const ShaderEvalContext &context) const {
    TextureSample sample = m_input->evaluate(context);
    sample.rgb *= m_scale;
    return sample;
}

TextureKind ScaleTexture::kind() const {
    return TextureKind::Scale;
}

const TextureHandle &ScaleTexture::input() const {
    return m_input;
}

double ScaleTexture::scale() const {
    return m_scale;
}

UVTransformTexture::UVTransformTexture(TextureHandle input,
                                       const vec2 &offset,
                                       const vec2 &scale,
                                       double rotation_radians)
    : m_input(std::move(input)), m_offset(offset), m_scale(scale),
      m_cos_rotation(std::cos(rotation_radians)),
      m_sin_rotation(std::sin(rotation_radians)) {
}

TextureSample
UVTransformTexture::evaluate(const ShaderEvalContext &context) const {
    ShaderEvalContext transformed = context;
    const double scaled_u = context.uv0.x() * m_scale.x();
    const double scaled_v = context.uv0.y() * m_scale.y();
    transformed.uv0 =
        vec2(m_offset.x() + m_cos_rotation * scaled_u -
                               m_sin_rotation * scaled_v,
             m_offset.y() + m_sin_rotation * scaled_u +
                               m_cos_rotation * scaled_v);
    return m_input->evaluate(transformed);
}

TextureKind UVTransformTexture::kind() const {
    return TextureKind::UVTransform;
}

const TextureHandle &UVTransformTexture::input() const {
    return m_input;
}

const vec2 &UVTransformTexture::offset() const {
    return m_offset;
}

const vec2 &UVTransformTexture::scale() const {
    return m_scale;
}

double UVTransformTexture::cos_rotation() const {
    return m_cos_rotation;
}

double UVTransformTexture::sin_rotation() const {
    return m_sin_rotation;
}

MultiplyTexture::MultiplyTexture(TextureHandle a, TextureHandle b)
    : m_a(std::move(a)), m_b(std::move(b)) {
}

TextureSample
MultiplyTexture::evaluate(const ShaderEvalContext &context) const {
    const TextureSample a = m_a->evaluate(context);
    const TextureSample b = m_b->evaluate(context);
    return TextureSample{a.rgb * b.rgb, a.alpha * b.alpha};
}

TextureKind MultiplyTexture::kind() const {
    return TextureKind::Multiply;
}

const TextureHandle &MultiplyTexture::a() const {
    return m_a;
}

const TextureHandle &MultiplyTexture::b() const {
    return m_b;
}

MixTexture::MixTexture(TextureHandle a, TextureHandle b,
                       TextureHandle factor)
    : m_a(std::move(a)), m_b(std::move(b)),
      m_factor(std::move(factor)) {
}

TextureSample MixTexture::evaluate(const ShaderEvalContext &context) const {
    const TextureSample a = m_a->evaluate(context);
    const TextureSample b = m_b->evaluate(context);
    const double factor =
        clamp(texture_scalar(m_factor, context), 0.0, 1.0);
    return TextureSample{lerp(a.rgb, b.rgb, factor),
                         (1.0 - factor) * a.alpha + factor * b.alpha};
}

TextureKind MixTexture::kind() const {
    return TextureKind::Mix;
}

const TextureHandle &MixTexture::a() const {
    return m_a;
}

const TextureHandle &MixTexture::b() const {
    return m_b;
}

const TextureHandle &MixTexture::factor() const {
    return m_factor;
}

ColorRampTexture::ColorRampTexture(TextureHandle input, const color &low,
                                   const color &high, double min_value,
                                   double max_value)
    : m_input(std::move(input)), m_low(low), m_high(high),
      m_min_value(min_value), m_max_value(max_value) {
}

TextureSample
ColorRampTexture::evaluate(const ShaderEvalContext &context) const {
    const double denominator = m_max_value - m_min_value;
    const double value = texture_scalar(m_input, context);
    const double factor =
        denominator == 0.0
            ? 0.0
            : clamp((value - m_min_value) / denominator, 0.0, 1.0);
    return TextureSample{lerp(m_low, m_high, factor), 1.0};
}

TextureKind ColorRampTexture::kind() const {
    return TextureKind::ColorRamp;
}

const TextureHandle &ColorRampTexture::input() const {
    return m_input;
}

const color &ColorRampTexture::low() const {
    return m_low;
}

const color &ColorRampTexture::high() const {
    return m_high;
}

double ColorRampTexture::min_value() const {
    return m_min_value;
}

double ColorRampTexture::max_value() const {
    return m_max_value;
}

#ifndef SCENE_IR_H
#define SCENE_IR_H

#include "json.hpp"
#include "normal_mapping.h"
#include "scene_config.h"
#include "scene_description.h"
#include "texture.h"

#include <cstddef>
#include <limits>
#include <string>
#include <variant>
#include <vector>

using TextureIRId = std::size_t;
constexpr TextureIRId kInvalidTextureIR =
    std::numeric_limits<TextureIRId>::max();

enum class TextureSemantic {
    Color,
    Scalar,
    Normal
};

enum class TextureColorSpaceIR {
    Auto,
    SRGB,
    Linear
};

struct ConstantTextureIR {
    color value{0, 0, 0};
};

struct AliasTextureIR {
    TextureIRId target = kInvalidTextureIR;
};

struct CheckerTextureIR {
    TextureIRId even = kInvalidTextureIR;
    TextureIRId odd = kInvalidTextureIR;
};

struct NoiseTextureIR {
    double scale = 1.0;
};

struct ImageTextureIR {
    std::string path;
    TextureColorSpaceIR color_space = TextureColorSpaceIR::Auto;
    TextureChannel channel = TextureChannel::RGB;
    bool channel_explicit = false;
    SamplerState sampler;
};

struct ScaleTextureIR {
    TextureIRId input = kInvalidTextureIR;
    double scale = 1.0;
};

struct MultiplyTextureIR {
    TextureIRId a = kInvalidTextureIR;
    TextureIRId b = kInvalidTextureIR;
};

struct MixTextureIR {
    TextureIRId a = kInvalidTextureIR;
    TextureIRId b = kInvalidTextureIR;
    TextureIRId factor = kInvalidTextureIR;
};

struct ColorRampTextureIR {
    TextureIRId input = kInvalidTextureIR;
    color low{0, 0, 0};
    color high{1, 1, 1};
    double min_value = 0.0;
    double max_value = 1.0;
};

using TextureIRData =
    std::variant<ConstantTextureIR, AliasTextureIR, CheckerTextureIR,
                 NoiseTextureIR, ImageTextureIR, ScaleTextureIR,
                 MultiplyTextureIR, MixTextureIR, ColorRampTextureIR>;

struct TextureIRNode {
    std::string name;
    TextureIRData data;
};

struct LambertianMaterialIR {
    TextureIRId albedo = kInvalidTextureIR;
};

struct MetalMaterialIR {
    color albedo{1, 1, 1};
    double roughness = 0.0;
};

struct DielectricMaterialIR {
    double ior = 1.5;
};

struct DiffuseLightMaterialIR {
    TextureIRId emission = kInvalidTextureIR;
};

struct PrincipledMaterialIR {
    TextureIRId base_color = kInvalidTextureIR;
    TextureIRId roughness = kInvalidTextureIR;
    TextureIRId metallic = kInvalidTextureIR;
    TextureIRId normal_map = kInvalidTextureIR;
    TextureIRId emission = kInvalidTextureIR;
    double emission_strength = 1.0;
    TextureIRId clearcoat = kInvalidTextureIR;
    TextureIRId clearcoat_roughness = kInvalidTextureIR;
    NormalMapSettings normal_settings;
};

using MaterialIRData =
    std::variant<LambertianMaterialIR, MetalMaterialIR,
                 DielectricMaterialIR, DiffuseLightMaterialIR,
                 PrincipledMaterialIR>;

struct MaterialIR {
    std::string name;
    MaterialIRData data;
};

struct SceneObjectSpec {
    std::string type;
    nlohmann::json data;
};

struct SceneIR {
    std::string source_path;
    std::string name;
    CameraConfig camera;
    RenderPreset preset;
    std::vector<TextureIRNode> textures;
    std::vector<MaterialIR> materials;
    std::vector<SceneObjectSpec> objects;
    std::vector<SceneObjectSpec> lights;
    bool world_accel = true;
    bool auto_emitters = true;
    double time0 = 0.0;
    double time1 = 1.0;
};

SceneIR parse_scene_ir(const SceneDescription &description);
SceneConfig build_scene_config(const SceneIR &ir);

#endif

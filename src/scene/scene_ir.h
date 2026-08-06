#ifndef SCENE_IR_H
#define SCENE_IR_H

#include "image_sampling.h"
#include "normal_map_types.h"
#include "scene_description.h"
#include "scene_types.h"
#include "transform.h"

#include <array>
#include <cstddef>
#include <limits>
#include <map>
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
    bool double_sided = true;
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

using ObjectIRId = std::size_t;
constexpr ObjectIRId kInvalidObjectIR =
    std::numeric_limits<ObjectIRId>::max();

struct SphereObjectIR {
    point3 center{0, 0, 0};
    double radius = 1.0;
    std::string material;
};

struct MovingSphereObjectIR {
    point3 center0{0, 0, 0};
    point3 center1{0, 0, 0};
    double time0 = 0.0;
    double time1 = 1.0;
    double radius = 1.0;
    std::string material;
};

struct BoxObjectIR {
    point3 minimum{0, 0, 0};
    point3 maximum{1, 1, 1};
    std::string material;
};

enum class AxisRectPlane {
    XY,
    XZ,
    YZ
};

struct AxisRectObjectIR {
    AxisRectPlane plane = AxisRectPlane::XY;
    double a0 = 0.0;
    double a1 = 1.0;
    double b0 = 0.0;
    double b1 = 1.0;
    double k = 0.0;
    std::string material;
};

struct QuadObjectIR {
    point3 origin{0, 0, 0};
    vec3 u{1, 0, 0};
    vec3 v{0, 1, 0};
    std::string material;
};

struct TriangleObjectIR {
    std::array<point3, 3> positions;
    std::array<vec3, 3> normals;
    std::array<vec2, 3> uv0;
    bool has_normals = false;
    bool has_uv0 = false;
    std::string material;
};

struct ObjObjectIR {
    std::string path;
    vec3 local_translation{0, 0, 0};
    vec3 scale{1, 1, 1};
    double rotation_y = 0.0;
    vec3 position{0, 0, 0};
    bool has_position = false;
    bool auto_lift_to_ground = false;
    bool build_bvh = true;
    bool use_vertex_normals = true;
    std::string material;
};

struct ModelObjectIR {
    std::string path;
    Transform transform;
    int scene_index = -1;
    std::map<std::string, std::string> material_overrides;
};

struct TransformObjectIR {
    ObjectIRId child = kInvalidObjectIR;
    Transform transform;
};

struct FlipFaceObjectIR {
    ObjectIRId child = kInvalidObjectIR;
};

struct ConstantMediumObjectIR {
    ObjectIRId boundary = kInvalidObjectIR;
    double density = 1.0;
    TextureIRId texture = kInvalidTextureIR;
    color albedo{1, 1, 1};
};

struct GroupObjectIR {
    std::vector<ObjectIRId> children;
    bool accelerate = false;
    double time0 = 0.0;
    double time1 = 1.0;
};

using ObjectIRData =
    std::variant<SphereObjectIR, MovingSphereObjectIR, BoxObjectIR,
                 AxisRectObjectIR, QuadObjectIR, TriangleObjectIR,
                 ObjObjectIR, ModelObjectIR, TransformObjectIR,
                 FlipFaceObjectIR, ConstantMediumObjectIR, GroupObjectIR>;

struct ObjectIRNode {
    std::string context;
    ObjectIRData data;
};

struct PointLightIR {
    point3 position{0, 0, 0};
    color intensity{1, 1, 1};
};

struct DirectionalLightIR {
    vec3 direction{0, -1, 0};
    color radiance{1, 1, 1};
};

struct SpotLightIR {
    point3 position{0, 0, 0};
    vec3 direction{0, -1, 0};
    double cutoff = 20.0;
    color intensity{1, 1, 1};
};

struct QuadLightIR {
    point3 origin{0, 0, 0};
    vec3 u{1, 0, 0};
    vec3 v{0, 1, 0};
    color intensity{1, 1, 1};
};

struct EnvironmentLightIR {
    std::string path;
};

using LightIRData =
    std::variant<PointLightIR, DirectionalLightIR, SpotLightIR, QuadLightIR,
                 EnvironmentLightIR>;

struct LightIR {
    std::string context;
    LightIRData data;
};

struct SceneIR {
    std::string source_path;
    std::string name;
    CameraConfig camera;
    RenderPreset preset;
    std::vector<TextureIRNode> textures;
    std::vector<MaterialIR> materials;
    std::vector<ObjectIRNode> object_nodes;
    std::vector<ObjectIRId> objects;
    std::vector<LightIR> lights;
    bool world_accel = true;
    bool auto_emitters = true;
    double time0 = 0.0;
    double time1 = 1.0;
};

SceneIR parse_scene_ir(const SceneDescription &description);
SceneIR load_scene_ir_file(const std::string &path);

#endif

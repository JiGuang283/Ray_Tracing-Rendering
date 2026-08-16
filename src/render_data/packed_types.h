#ifndef PACKED_TYPES_H
#define PACKED_TYPES_H

#include "host_device.h"
#include "integrator_policy.h"

#include <cstdint>
#include <limits>
#include <type_traits>

constexpr std::uint32_t kInvalidPackedIndex =
    std::numeric_limits<std::uint32_t>::max();
constexpr std::uint32_t kPackedTextureStackCapacity = 16;
constexpr std::uint32_t kPackedTextureValueCapacity =
    2 * kPackedTextureStackCapacity;

template <typename Tag> struct Handle32 {
    std::uint32_t value = kInvalidPackedIndex;

    RT_HOST_DEVICE constexpr bool valid() const noexcept {
        return value != kInvalidPackedIndex;
    }
};

struct MeshTag;
struct InstanceTag;
struct TransformTag;
struct AggregateTag;
struct MaterialTag;
struct TextureTag;
struct ImageTag;
struct PerlinTag;

using MeshId = Handle32<MeshTag>;
using InstanceId = Handle32<InstanceTag>;
using TransformId = Handle32<TransformTag>;
using AggregateId = Handle32<AggregateTag>;
using MaterialId = Handle32<MaterialTag>;
using TextureId = Handle32<TextureTag>;
using ImageId = Handle32<ImageTag>;
using PerlinId = Handle32<PerlinTag>;

struct Range32 {
    std::uint32_t offset = 0;
    std::uint32_t count = 0;
};

struct Float2 {
    float x = 0.0f;
    float y = 0.0f;
};

struct Float3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct Float4 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 0.0f;
};

enum class PackedGeometryType : std::uint32_t {
    Mesh = 0,
    Sphere = 1,
    MovingSphere = 2,
    Medium = 3
};

enum class PackedMaterialType : std::uint32_t {
    Lambertian = 0,
    Metal = 1,
    Dielectric = 2,
    DiffuseLight = 3,
    Principled = 4,
    Isotropic = 5
};

enum class PackedTextureType : std::uint32_t {
    Constant = 0,
    VertexColor = 1,
    Checker = 2,
    Noise = 3,
    Image = 4,
    Scale = 5,
    UVTransform = 6,
    Multiply = 7,
    Mix = 8,
    ColorRamp = 9
};

enum class PackedLightType : std::uint32_t {
    Point = 0,
    Directional = 1,
    Spot = 2,
    Quad = 3,
    Environment = 4,
    SphereEmitter = 5,
    TriangleEmitter = 6,
    MeshEmitter = 7
};

enum class PackedClosureType : std::uint32_t {
    Lambertian = 0,
    Mirror = 1,
    Dielectric = 2,
    GGXReflection = 3,
    ClearcoatGGX = 4,
    IsotropicPhase = 5
};

enum class PackedShadingStatus : std::uint32_t {
    Success = 0,
    Miss = 1,
    InvalidInput = 2,
    InvalidMaterial = 3,
    TextureFailure = 4,
    TextureStackOverflow = 5,
    ClosureOverflow = 6,
    NonFinite = 7
};

enum class PackedBSDFStatus : std::uint32_t {
    Success = 0,
    Empty = 1,
    InvalidInput = 2,
    NoSample = 3,
    NonFinite = 4
};

enum class PackedLightStatus : std::uint32_t {
    Success = 0,
    NoSample = 1,
    InvalidInput = 2,
    InvalidDistribution = 3,
    TextureFailure = 4,
    NonFinite = 5
};

enum class PackedTransportStatus : std::uint32_t {
    Success = 0,
    InvalidInput = 1,
    TraversalFailure = 2,
    ReconstructionFailure = 3,
    MaterialFailure = 4,
    LightFailure = 5,
    BSDFFailure = 6,
    NonFinite = 7
};

enum PackedPathFlags : std::uint32_t {
    PACKED_PATH_NONE = 0,
    PACKED_PATH_ACTIVE = 1u << 0,
    PACKED_PATH_DELTA_BOUNCE = 1u << 1
};

enum PackedInstanceFlags : std::uint32_t {
    PACKED_INSTANCE_NONE = 0,
    PACKED_INSTANCE_FLIP_FACE = 1u << 0,
    // Only the host fast transport path consumes this bit. Safe CUDA kernels
    // deliberately ignore it, preserving bit-identical device behavior.
    PACKED_INSTANCE_HOST_IDENTITY_TRANSFORM = 1u << 1
};

enum PackedTriangleFlags : std::uint32_t {
    PACKED_TRIANGLE_NONE = 0,
    PACKED_TRIANGLE_HAS_NORMALS = 1u << 0,
    PACKED_TRIANGLE_HAS_UV = 1u << 1,
    PACKED_TRIANGLE_HAS_COLOR = 1u << 2,
    PACKED_TRIANGLE_HAS_TANGENT = 1u << 3,
    PACKED_TRIANGLE_REVERSE_EMITTER_NORMAL = 1u << 4
};

enum PackedSphereFlags : std::uint32_t {
    PACKED_SPHERE_NONE = 0,
    PACKED_SPHERE_FLIP_ORIENTATION = 1u << 0
};

enum PackedHitFlags : std::uint32_t {
    PACKED_HIT_NONE = 0,
    PACKED_HIT_FRONT_FACE = 1u << 0,
    PACKED_HIT_TRIANGLE = 1u << 1,
    PACKED_HIT_SPHERE = 1u << 2,
    PACKED_HIT_MEDIUM = 1u << 3
};

enum PackedClosureFlags : std::uint32_t {
    PACKED_CLOSURE_NONE = 0,
    PACKED_CLOSURE_FRONT_FACE = 1u << 0
};

enum PackedBSDFSampleFlags : std::uint32_t {
    PACKED_BSDF_NONE = 0,
    PACKED_BSDF_DIFFUSE = 1u << 0,
    PACKED_BSDF_GLOSSY = 1u << 1,
    PACKED_BSDF_DELTA = 1u << 2,
    PACKED_BSDF_REFLECTION = 1u << 3,
    PACKED_BSDF_TRANSMISSION = 1u << 4,
    PACKED_BSDF_PHASE = 1u << 5
};

enum class PackedTraversalStatus : std::uint32_t {
    Miss = 0,
    Hit = 1,
    StackOverflow = 2,
    InvalidInput = 3
};

enum PackedMaterialFlags : std::uint32_t {
    PACKED_MATERIAL_NONE = 0,
    PACKED_MATERIAL_EMISSIVE = 1u << 0,
    PACKED_MATERIAL_DOUBLE_SIDED = 1u << 1,
    PACKED_MATERIAL_NORMAL_DIRECTX = 1u << 2
};

enum PackedLightFlags : std::uint32_t {
    PACKED_LIGHT_NONE = 0,
    PACKED_LIGHT_DELTA = 1u << 0,
    PACKED_LIGHT_INFINITE = 1u << 1,
    PACKED_LIGHT_BSDF_HITTABLE = 1u << 2,
    PACKED_LIGHT_DOUBLE_SIDED = 1u << 3,
    PACKED_LIGHT_ENVIRONMENT_PROBE = 1u << 4,
    PACKED_LIGHT_ENVIRONMENT_SRGB = 1u << 5
};

enum PackedImageFlags : std::uint32_t {
    PACKED_IMAGE_NONE = 0,
    PACKED_IMAGE_HDR = 1u << 0
};

enum PackedSamplerFlags : std::uint32_t {
    PACKED_SAMPLER_WRAP_U_SHIFT = 0,
    PACKED_SAMPLER_WRAP_V_SHIFT = 2,
    PACKED_SAMPLER_FILTER_SHIFT = 4,
    PACKED_SAMPLER_FLIP_V = 1u << 6,
    PACKED_SAMPLER_SRGB = 1u << 7
};

enum PackedBVHMeta : std::uint32_t {
    PACKED_BVH_LEAF_BIT = 0x80000000u,
    PACKED_BVH_VALUE_MASK = 0x7fffffffu
};

struct alignas(16) PackedBVHNode {
    Float3 bounds_min;
    std::uint32_t first = 0;
    Float3 bounds_max;
    std::uint32_t meta = 0;

    RT_HOST_DEVICE bool is_leaf() const noexcept {
        return (meta & PACKED_BVH_LEAF_BIT) != 0;
    }

    RT_HOST_DEVICE std::uint32_t primitive_count() const noexcept {
        return meta & PACKED_BVH_VALUE_MASK;
    }
};

struct alignas(16) PackedRay {
    Float3 origin;
    float t_min = 0.0f;
    Float3 direction;
    float t_max = std::numeric_limits<float>::infinity();
    float time = 0.0f;
    std::uint32_t padding[3]{};
};

struct alignas(16) PackedHit {
    float t = std::numeric_limits<float>::infinity();
    float barycentric_u = 0.0f;
    float barycentric_v = 0.0f;
    std::uint32_t instance_id = kInvalidPackedIndex;
    std::uint32_t primitive_id = kInvalidPackedIndex;
    std::uint32_t flags = 0;
    std::uint32_t padding[2]{};
};

struct alignas(16) PackedSurfaceInteraction {
    Float3 position;
    std::uint32_t material_id = kInvalidPackedIndex;
    Float3 geometric_normal;
    std::uint32_t instance_id = kInvalidPackedIndex;
    Float3 shading_normal;
    std::uint32_t primitive_id = kInvalidPackedIndex;
    Float3 dpdu;
    std::uint32_t flags = 0;
    Float3 dpdv;
    float vertex_alpha = 1.0f;
    Float2 uv;
    std::uint32_t emitter_id = kInvalidPackedIndex;
    std::uint32_t padding = 0;
    Float4 vertex_color{1.0f, 1.0f, 1.0f, 1.0f};
};

struct alignas(16) PackedShadingFrame {
    Float3 tangent{1.0f, 0.0f, 0.0f};
    float handedness = 1.0f;
    Float3 normal{0.0f, 0.0f, 1.0f};
    float padding = 0.0f;
};

struct alignas(16) PackedClosure {
    Float4 parameters{};
    PackedClosureType type = PackedClosureType::Lambertian;
    std::uint32_t flags = PACKED_CLOSURE_NONE;
    float contribution_weight = 1.0f;
    float sample_weight = 1.0f;
};

struct alignas(16) PackedMaterialOutput {
    static constexpr std::uint32_t kMaxClosures = 8;

    PackedShadingFrame frame;
    Float3 geometry_normal{0.0f, 0.0f, 1.0f};
    std::uint32_t closure_count = 0;
    Float3 emission{};
    float opacity = 1.0f;
    PackedClosure closures[kMaxClosures]{};
};

struct alignas(16) PackedBSDFSample {
    Float3 wi{};
    float pdf = 0.0f;
    Float3 f{};
    float eta = 1.0f;
    std::uint32_t flags = PACKED_BSDF_NONE;
    std::uint32_t closure_index = kInvalidPackedIndex;
    std::uint32_t padding[2]{};

    RT_HOST_DEVICE bool is_delta() const noexcept {
        return (flags & PACKED_BSDF_DELTA) != 0;
    }

    RT_HOST_DEVICE bool is_transmission() const noexcept {
        return (flags & PACKED_BSDF_TRANSMISSION) != 0;
    }

    RT_HOST_DEVICE bool is_phase() const noexcept {
        return (flags & PACKED_BSDF_PHASE) != 0;
    }
};

struct alignas(16) PackedLightSample {
    Float3 wi{0.0f, 0.0f, 1.0f};
    float distance = std::numeric_limits<float>::infinity();
    Float3 radiance{};
    float pdf = 0.0f;
    std::uint32_t light_id = kInvalidPackedIndex;
    std::uint32_t flags = PACKED_LIGHT_NONE;
    std::uint32_t element_id = kInvalidPackedIndex;
    std::uint32_t padding = 0;

    RT_HOST_DEVICE bool is_delta() const noexcept {
        return (flags & PACKED_LIGHT_DELTA) != 0;
    }

    RT_HOST_DEVICE bool is_infinite() const noexcept {
        return (flags & PACKED_LIGHT_INFINITE) != 0;
    }

    RT_HOST_DEVICE bool is_bsdf_hittable() const noexcept {
        return (flags & PACKED_LIGHT_BSDF_HITTABLE) != 0;
    }
};

struct alignas(16) SelectedPackedLightSample {
    PackedLightSample sample;
    float selection_probability = 0.0f;
    std::uint32_t selection_index = kInvalidPackedIndex;
    std::uint32_t padding[2]{};

    RT_HOST_DEVICE float combined_pdf() const noexcept {
        return sample.pdf * selection_probability;
    }
};

struct alignas(16) PackedTransportSettings {
    IntegratorPolicy policy{};
    std::uint32_t max_depth = 50;
    std::uint32_t padding[3]{};
};

struct alignas(16) PackedTransportResult {
    Float3 radiance{};
    PackedTransportStatus status = PackedTransportStatus::Success;
    std::uint32_t depth = 0;
    std::uint32_t shadow_rays = 0;
    std::uint32_t traversal_steps = 0;
    std::uint32_t padding = 0;
};

struct alignas(16) PackedPathState {
    PackedRay ray;
    Float3 throughput{1.0f, 1.0f, 1.0f};
    float eta_scale = 1.0f;
    Float3 radiance{};
    float previous_bsdf_pdf = 0.0f;
    std::uint32_t rng_state = 1;
    std::uint32_t pixel_index = 0;
    std::uint32_t sample_index = 0;
    std::uint32_t depth = 0;
    PackedTransportStatus status = PackedTransportStatus::Success;
    std::uint32_t flags = PACKED_PATH_NONE;
    std::uint32_t shadow_rays = 0;
    std::uint32_t traversal_steps = 0;

    RT_HOST_DEVICE bool active() const noexcept {
        return (flags & PACKED_PATH_ACTIVE) != 0;
    }

    RT_HOST_DEVICE bool delta_bounce() const noexcept {
        return (flags & PACKED_PATH_DELTA_BOUNCE) != 0;
    }
};

struct alignas(16) PackedTransform {
    float object_to_world[12]{};
    float world_to_object[12]{};
};

struct PackedTriangle {
    std::uint32_t vertex0 = 0;
    std::uint32_t vertex1 = 0;
    std::uint32_t vertex2 = 0;
    std::uint32_t material_slot = 0;
    std::uint32_t primitive_id = 0;
    std::uint32_t flags = 0;
};

struct alignas(16) PackedSphere {
    Float3 center;
    float radius = 1.0f;
    std::uint32_t flags = 0;
    std::uint32_t padding[3]{};
};

struct alignas(16) PackedMovingSphere {
    Float3 center0;
    float time0 = 0.0f;
    Float3 center1;
    float time1 = 1.0f;
    float radius = 1.0f;
    std::uint32_t flags = 0;
    std::uint32_t padding[2]{};
};

struct alignas(16) PackedMesh {
    Range32 vertices;
    Range32 triangles;
    Range32 bvh_nodes;
    std::uint32_t material_slot_count = 0;
    std::uint32_t flags = 0;
    Float3 bounds_min;
    std::uint32_t padding0 = 0;
    Float3 bounds_max;
    std::uint32_t padding1 = 0;
};

struct alignas(16) PackedInstance {
    PackedGeometryType geometry_type = PackedGeometryType::Mesh;
    std::uint32_t geometry_index = kInvalidPackedIndex;
    std::uint32_t transform_id = kInvalidPackedIndex;
    std::uint32_t flags = 0;
    Range32 material_bindings;
    std::uint32_t source_object_id = kInvalidPackedIndex;
    std::uint32_t padding0 = 0;
    Float3 bounds_min;
    std::uint32_t padding1 = 0;
    Float3 bounds_max;
    std::uint32_t padding2 = 0;
};

struct PackedAggregate {
    Range32 bvh_nodes;
    Range32 instance_indices;
};

struct PackedMedium {
    std::uint32_t boundary_aggregate = kInvalidPackedIndex;
    std::uint32_t phase_material = kInvalidPackedIndex;
    float neg_inv_density = -1.0f;
    std::uint32_t flags = 0;
};

struct alignas(16) PackedTextureNode {
    PackedTextureType type = PackedTextureType::Constant;
    std::uint32_t input0 = kInvalidPackedIndex;
    std::uint32_t input1 = kInvalidPackedIndex;
    std::uint32_t input2 = kInvalidPackedIndex;
    std::uint32_t image_id = kInvalidPackedIndex;
    std::uint32_t perlin_id = kInvalidPackedIndex;
    std::uint32_t sampler_flags = 0;
    std::uint32_t channel = 0;
    Float4 value0;
    Float4 value1;
    Float4 value2;
};

struct alignas(16) PackedMaterial {
    PackedMaterialType type = PackedMaterialType::Lambertian;
    std::uint32_t flags = 0;
    std::uint32_t texture_ids[8]{
        kInvalidPackedIndex, kInvalidPackedIndex, kInvalidPackedIndex,
        kInvalidPackedIndex, kInvalidPackedIndex, kInvalidPackedIndex,
        kInvalidPackedIndex, kInvalidPackedIndex};
    std::uint32_t padding[2]{};
    Float4 parameters[4]{};
    Float4 emission_estimate{};
};

struct PackedImageDesc {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t channels = 0;
    std::uint32_t flags = 0;
    Range32 texels;
};

struct PackedPerlinDesc {
    Range32 gradients;
    Range32 permutation_x;
    Range32 permutation_y;
    Range32 permutation_z;
};

struct alignas(16) PackedLight {
    PackedLightType type = PackedLightType::Point;
    std::uint32_t flags = 0;
    std::uint32_t instance_id = kInvalidPackedIndex;
    std::uint32_t material_id = kInvalidPackedIndex;
    Range32 distribution;
    std::uint32_t image_id = kInvalidPackedIndex;
    float selection_probability = 0.0f;
    Range32 element_indices;
    std::uint32_t padding1[2]{};
    Float4 data0;
    Float4 data1;
    Float4 data2;
    Float4 radiance;
    Float4 power;
};

struct alignas(16) PackedCamera {
    Float3 origin;
    float lens_radius = 0.0f;
    Float3 lower_left_corner;
    float time0 = 0.0f;
    Float3 horizontal;
    float time1 = 0.0f;
    Float3 vertical;
    float padding = 0.0f;
};

static_assert(sizeof(Float2) == 8);
static_assert(sizeof(Float3) == 12);
static_assert(sizeof(Float4) == 16);
static_assert(sizeof(PackedBVHNode) == 32);
static_assert(sizeof(PackedRay) == 48);
static_assert(sizeof(PackedHit) == 32);
static_assert(sizeof(PackedSurfaceInteraction) == 112);
static_assert(sizeof(PackedShadingFrame) == 32);
static_assert(sizeof(PackedClosure) == 32);
static_assert(sizeof(PackedMaterialOutput) == 320);
static_assert(sizeof(PackedBSDFSample) == 48);
static_assert(sizeof(PackedLightSample) == 48);
static_assert(sizeof(SelectedPackedLightSample) == 64);
static_assert(sizeof(PackedTransportSettings) == 32);
static_assert(sizeof(PackedTransportResult) == 32);
static_assert(sizeof(PackedPathState) == 112);
static_assert(sizeof(PackedTransform) == 96);
static_assert(std::is_trivially_copyable_v<PackedBVHNode>);
static_assert(std::is_trivially_copyable_v<Range32>);
static_assert(std::is_trivially_copyable_v<PackedRay>);
static_assert(std::is_trivially_copyable_v<PackedHit>);
static_assert(std::is_trivially_copyable_v<PackedTraversalStatus>);
static_assert(std::is_trivially_copyable_v<PackedShadingStatus>);
static_assert(std::is_trivially_copyable_v<PackedBSDFStatus>);
static_assert(std::is_trivially_copyable_v<PackedLightStatus>);
static_assert(std::is_trivially_copyable_v<IntegratorPolicy>);
static_assert(std::is_trivially_copyable_v<PackedTransportStatus>);
static_assert(std::is_trivially_copyable_v<PackedSurfaceInteraction>);
static_assert(std::is_trivially_copyable_v<PackedShadingFrame>);
static_assert(std::is_trivially_copyable_v<PackedClosure>);
static_assert(std::is_trivially_copyable_v<PackedMaterialOutput>);
static_assert(std::is_trivially_copyable_v<PackedBSDFSample>);
static_assert(std::is_trivially_copyable_v<PackedLightSample>);
static_assert(std::is_trivially_copyable_v<SelectedPackedLightSample>);
static_assert(std::is_trivially_copyable_v<PackedTransportSettings>);
static_assert(std::is_trivially_copyable_v<PackedTransportResult>);
static_assert(std::is_trivially_copyable_v<PackedPathState>);
static_assert(std::is_trivially_copyable_v<PackedTransform>);
static_assert(std::is_trivially_copyable_v<PackedTriangle>);
static_assert(std::is_trivially_copyable_v<PackedSphere>);
static_assert(std::is_trivially_copyable_v<PackedMovingSphere>);
static_assert(std::is_trivially_copyable_v<PackedMesh>);
static_assert(std::is_trivially_copyable_v<PackedInstance>);
static_assert(std::is_trivially_copyable_v<PackedAggregate>);
static_assert(std::is_trivially_copyable_v<PackedMedium>);
static_assert(std::is_trivially_copyable_v<PackedTextureNode>);
static_assert(std::is_trivially_copyable_v<PackedMaterial>);
static_assert(std::is_trivially_copyable_v<PackedImageDesc>);
static_assert(std::is_trivially_copyable_v<PackedPerlinDesc>);
static_assert(std::is_trivially_copyable_v<PackedLight>);
static_assert(std::is_trivially_copyable_v<PackedCamera>);

#endif

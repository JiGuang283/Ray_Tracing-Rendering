#ifndef COMPILED_SCENE_H
#define COMPILED_SCENE_H

#include "packed_types.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

template <typename T> struct PackedArrayView {
    const T *data = nullptr;
    std::uint32_t count = 0;

    RT_HOST_DEVICE const T &operator[](std::uint32_t index) const noexcept {
        return data[index];
    }
};

struct CompiledScene {
    PackedCamera camera;
    Float3 background;
    float scene_time0 = 0.0f;
    float scene_time1 = 1.0f;

    std::vector<Float4> positions;
    std::vector<Float4> normals;
    std::vector<Float4> tangents;
    std::vector<Float2> uv0;
    std::vector<Float4> vertex_colors;
    std::vector<PackedTriangle> triangles;
    std::vector<PackedMesh> meshes;
    std::vector<PackedSphere> spheres;
    std::vector<PackedMovingSphere> moving_spheres;
    std::vector<PackedTransform> transforms;
    std::vector<PackedInstance> instances;
    std::vector<std::uint32_t> material_bindings;
    std::vector<std::uint32_t> emitter_bindings;
    std::vector<PackedAggregate> aggregates;
    std::vector<std::uint32_t> aggregate_instance_indices;
    std::vector<PackedBVHNode> bvh_nodes;
    std::vector<PackedMedium> media;

    std::vector<PackedMaterial> materials;
    std::vector<PackedTextureNode> texture_nodes;
    std::vector<PackedImageDesc> images;
    std::vector<float> image_texels;
    std::vector<PackedPerlinDesc> perlin_tables;
    std::vector<Float4> perlin_gradients;
    std::vector<std::uint32_t> perlin_permutations;

    std::vector<PackedLight> lights;
    std::vector<std::uint32_t> delta_light_indices;
    std::vector<std::uint32_t> non_delta_light_indices;
    std::vector<float> light_selection_probabilities;
    std::vector<float> light_cdf;
    std::vector<std::uint32_t> light_element_indices;
    std::vector<float> light_distributions;
};

struct CompiledSceneView {
    PackedCamera camera;
    Float3 background;
    float scene_time0 = 0.0f;
    float scene_time1 = 1.0f;

    PackedArrayView<Float4> positions;
    PackedArrayView<Float4> normals;
    PackedArrayView<Float4> tangents;
    PackedArrayView<Float2> uv0;
    PackedArrayView<Float4> vertex_colors;
    PackedArrayView<PackedTriangle> triangles;
    PackedArrayView<PackedMesh> meshes;
    PackedArrayView<PackedSphere> spheres;
    PackedArrayView<PackedMovingSphere> moving_spheres;
    PackedArrayView<PackedTransform> transforms;
    PackedArrayView<PackedInstance> instances;
    PackedArrayView<std::uint32_t> material_bindings;
    PackedArrayView<std::uint32_t> emitter_bindings;
    PackedArrayView<PackedAggregate> aggregates;
    PackedArrayView<std::uint32_t> aggregate_instance_indices;
    PackedArrayView<PackedBVHNode> bvh_nodes;
    PackedArrayView<PackedMedium> media;
    PackedArrayView<PackedMaterial> materials;
    PackedArrayView<PackedTextureNode> texture_nodes;
    PackedArrayView<PackedImageDesc> images;
    PackedArrayView<float> image_texels;
    PackedArrayView<PackedPerlinDesc> perlin_tables;
    PackedArrayView<Float4> perlin_gradients;
    PackedArrayView<std::uint32_t> perlin_permutations;
    PackedArrayView<PackedLight> lights;
    PackedArrayView<std::uint32_t> delta_light_indices;
    PackedArrayView<std::uint32_t> non_delta_light_indices;
    PackedArrayView<float> light_selection_probabilities;
    PackedArrayView<float> light_cdf;
    PackedArrayView<std::uint32_t> light_element_indices;
    PackedArrayView<float> light_distributions;
};

static_assert(std::is_trivially_copyable_v<CompiledSceneView>);

struct CompiledSceneStats {
    std::uint64_t bytes = 0;
    std::uint32_t vertices = 0;
    std::uint32_t triangles = 0;
    std::uint32_t bvh_nodes = 0;
    std::uint32_t meshes = 0;
    std::uint32_t instances = 0;
    std::uint32_t materials = 0;
    std::uint32_t textures = 0;
    std::uint32_t images = 0;
    std::uint32_t lights = 0;
};

struct ValidationReport {
    std::vector<std::string> errors;

    bool ok() const noexcept {
        return errors.empty();
    }
};

CompiledSceneView make_scene_view(const CompiledScene &scene);
CompiledSceneStats compiled_scene_stats(const CompiledScene &scene);
ValidationReport validate_compiled_scene(const CompiledScene &scene);

#endif

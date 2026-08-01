#ifndef DEVICE_SCENE_H
#define DEVICE_SCENE_H

#include "device_buffer.h"
#include "render_data/compiled_scene.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

namespace cuda_backend {

struct DeviceSceneView {
    CompiledSceneView scene;
};

static_assert(std::is_trivially_copyable_v<DeviceSceneView>);

struct DeviceMemoryInfo {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
};

struct DeviceSceneUploadStats {
    std::size_t bytes = 0;
    std::size_t free_bytes_before = 0;
    float milliseconds = 0.0f;
};

bool cuda_device_available(std::string *reason = nullptr);
std::string cuda_device_name();
DeviceMemoryInfo query_device_memory();

class DeviceSceneStorage {
public:
    DeviceSceneStorage() = default;
    ~DeviceSceneStorage() = default;

    DeviceSceneStorage(const DeviceSceneStorage &) = delete;
    DeviceSceneStorage &operator=(const DeviceSceneStorage &) = delete;
    DeviceSceneStorage(DeviceSceneStorage &&) noexcept = default;
    DeviceSceneStorage &operator=(DeviceSceneStorage &&) noexcept = default;

    DeviceSceneUploadStats upload(const CompiledScene &scene);
    void reset() noexcept;
    DeviceSceneView view() const noexcept;
    std::size_t allocated_bytes() const noexcept;

private:
    PackedCamera m_camera{};
    Float3 m_background{};
    float m_scene_time0 = 0.0f;
    float m_scene_time1 = 1.0f;

    DeviceBuffer<Float4> m_positions;
    DeviceBuffer<Float4> m_normals;
    DeviceBuffer<Float4> m_tangents;
    DeviceBuffer<Float2> m_uv0;
    DeviceBuffer<Float4> m_vertex_colors;
    DeviceBuffer<PackedTriangle> m_triangles;
    DeviceBuffer<PackedMesh> m_meshes;
    DeviceBuffer<PackedSphere> m_spheres;
    DeviceBuffer<PackedMovingSphere> m_moving_spheres;
    DeviceBuffer<PackedTransform> m_transforms;
    DeviceBuffer<PackedInstance> m_instances;
    DeviceBuffer<std::uint32_t> m_material_bindings;
    DeviceBuffer<PackedAggregate> m_aggregates;
    DeviceBuffer<std::uint32_t> m_aggregate_instance_indices;
    DeviceBuffer<PackedBVHNode> m_bvh_nodes;
    DeviceBuffer<PackedMedium> m_media;
    DeviceBuffer<PackedMaterial> m_materials;
    DeviceBuffer<PackedTextureNode> m_texture_nodes;
    DeviceBuffer<PackedImageDesc> m_images;
    DeviceBuffer<float> m_image_texels;
    DeviceBuffer<PackedPerlinDesc> m_perlin_tables;
    DeviceBuffer<Float4> m_perlin_gradients;
    DeviceBuffer<std::uint32_t> m_perlin_permutations;
    DeviceBuffer<PackedLight> m_lights;
    DeviceBuffer<std::uint32_t> m_delta_light_indices;
    DeviceBuffer<std::uint32_t> m_non_delta_light_indices;
    DeviceBuffer<float> m_light_selection_probabilities;
    DeviceBuffer<float> m_light_cdf;
    DeviceBuffer<std::uint32_t> m_light_element_indices;
    DeviceBuffer<float> m_light_distributions;
};

} // namespace cuda_backend

#endif

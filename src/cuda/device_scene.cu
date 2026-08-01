#include "device_scene.h"

#include <chrono>
#include <sstream>
#include <stdexcept>

namespace cuda_backend {
namespace {

template <typename T>
PackedArrayView<T> view_of(const DeviceBuffer<T> &buffer) noexcept {
    return {buffer.data(), buffer.size()};
}

std::string validation_message(const ValidationReport &report) {
    std::ostringstream message;
    message << "cannot upload invalid compiled scene";
    const std::size_t limit = report.errors.size() < 4 ? report.errors.size()
                                                       : 4;
    for (std::size_t index = 0; index < limit; ++index) {
        message << "\n  - " << report.errors[index];
    }
    if (report.errors.size() > limit) {
        message << "\n  - ... " << report.errors.size() - limit
                << " more errors";
    }
    return message.str();
}

} // namespace

bool cuda_device_available(std::string *reason) {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status == cudaSuccess && count > 0) {
        return true;
    }
    if (reason != nullptr) {
        *reason = status == cudaSuccess ? "no CUDA device found"
                                        : cudaGetErrorString(status);
    }
    cudaGetLastError();
    return false;
}

std::string cuda_device_name() {
    int device = 0;
    RT_CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    RT_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    return properties.name;
}

DeviceMemoryInfo query_device_memory() {
    DeviceMemoryInfo result;
    RT_CUDA_CHECK(cudaMemGetInfo(&result.free_bytes, &result.total_bytes));
    return result;
}

DeviceSceneUploadStats DeviceSceneStorage::upload(
    const CompiledScene &scene) {
    const ValidationReport validation = validate_compiled_scene(scene);
    if (!validation.ok()) {
        throw std::runtime_error(validation_message(validation));
    }

    const CompiledSceneStats scene_stats = compiled_scene_stats(scene);
    const DeviceMemoryInfo memory = query_device_memory();
    constexpr std::size_t reserve_bytes = 64ull * 1024ull * 1024ull;
    const std::size_t usable = memory.free_bytes > reserve_bytes
                                   ? memory.free_bytes - reserve_bytes
                                   : 0;
    if (scene_stats.bytes > usable) {
        std::ostringstream message;
        message << "compiled scene requires " << scene_stats.bytes
                << " device bytes, but only " << memory.free_bytes
                << " bytes are free";
        throw std::runtime_error(message.str());
    }

    const auto begin = std::chrono::steady_clock::now();
    DeviceSceneStorage staged;
    staged.m_camera = scene.camera;
    staged.m_background = scene.background;
    staged.m_scene_time0 = scene.scene_time0;
    staged.m_scene_time1 = scene.scene_time1;

#define UPLOAD_BUFFER(field) staged.m_##field.upload(scene.field)
    UPLOAD_BUFFER(positions);
    UPLOAD_BUFFER(normals);
    UPLOAD_BUFFER(tangents);
    UPLOAD_BUFFER(uv0);
    UPLOAD_BUFFER(vertex_colors);
    UPLOAD_BUFFER(triangles);
    UPLOAD_BUFFER(meshes);
    UPLOAD_BUFFER(spheres);
    UPLOAD_BUFFER(moving_spheres);
    UPLOAD_BUFFER(transforms);
    UPLOAD_BUFFER(instances);
    UPLOAD_BUFFER(material_bindings);
    UPLOAD_BUFFER(emitter_bindings);
    UPLOAD_BUFFER(aggregates);
    UPLOAD_BUFFER(aggregate_instance_indices);
    UPLOAD_BUFFER(bvh_nodes);
    UPLOAD_BUFFER(media);
    UPLOAD_BUFFER(materials);
    UPLOAD_BUFFER(texture_nodes);
    UPLOAD_BUFFER(images);
    UPLOAD_BUFFER(image_texels);
    UPLOAD_BUFFER(perlin_tables);
    UPLOAD_BUFFER(perlin_gradients);
    UPLOAD_BUFFER(perlin_permutations);
    UPLOAD_BUFFER(lights);
    UPLOAD_BUFFER(delta_light_indices);
    UPLOAD_BUFFER(non_delta_light_indices);
    UPLOAD_BUFFER(light_selection_probabilities);
    UPLOAD_BUFFER(light_cdf);
    UPLOAD_BUFFER(light_element_indices);
    UPLOAD_BUFFER(light_distributions);
#undef UPLOAD_BUFFER

    if (staged.allocated_bytes() != scene_stats.bytes) {
        throw std::logic_error("CUDA scene upload byte count mismatch");
    }
    *this = std::move(staged);
    const auto end = std::chrono::steady_clock::now();
    DeviceSceneUploadStats result;
    result.bytes = allocated_bytes();
    result.free_bytes_before = memory.free_bytes;
    result.milliseconds = std::chrono::duration<float, std::milli>(end - begin)
                              .count();
    return result;
}

void DeviceSceneStorage::reset() noexcept {
    *this = DeviceSceneStorage{};
}

DeviceSceneView DeviceSceneStorage::view() const noexcept {
    DeviceSceneView result;
    CompiledSceneView &scene = result.scene;
    scene.camera = m_camera;
    scene.background = m_background;
    scene.scene_time0 = m_scene_time0;
    scene.scene_time1 = m_scene_time1;
    scene.positions = view_of(m_positions);
    scene.normals = view_of(m_normals);
    scene.tangents = view_of(m_tangents);
    scene.uv0 = view_of(m_uv0);
    scene.vertex_colors = view_of(m_vertex_colors);
    scene.triangles = view_of(m_triangles);
    scene.meshes = view_of(m_meshes);
    scene.spheres = view_of(m_spheres);
    scene.moving_spheres = view_of(m_moving_spheres);
    scene.transforms = view_of(m_transforms);
    scene.instances = view_of(m_instances);
    scene.material_bindings = view_of(m_material_bindings);
    scene.emitter_bindings = view_of(m_emitter_bindings);
    scene.aggregates = view_of(m_aggregates);
    scene.aggregate_instance_indices =
        view_of(m_aggregate_instance_indices);
    scene.bvh_nodes = view_of(m_bvh_nodes);
    scene.media = view_of(m_media);
    scene.materials = view_of(m_materials);
    scene.texture_nodes = view_of(m_texture_nodes);
    scene.images = view_of(m_images);
    scene.image_texels = view_of(m_image_texels);
    scene.perlin_tables = view_of(m_perlin_tables);
    scene.perlin_gradients = view_of(m_perlin_gradients);
    scene.perlin_permutations = view_of(m_perlin_permutations);
    scene.lights = view_of(m_lights);
    scene.delta_light_indices = view_of(m_delta_light_indices);
    scene.non_delta_light_indices = view_of(m_non_delta_light_indices);
    scene.light_selection_probabilities =
        view_of(m_light_selection_probabilities);
    scene.light_cdf = view_of(m_light_cdf);
    scene.light_element_indices = view_of(m_light_element_indices);
    scene.light_distributions = view_of(m_light_distributions);
    return result;
}

std::size_t DeviceSceneStorage::allocated_bytes() const noexcept {
    std::size_t total = 0;
#define ADD_BUFFER_BYTES(field) total += m_##field.bytes()
    ADD_BUFFER_BYTES(positions);
    ADD_BUFFER_BYTES(normals);
    ADD_BUFFER_BYTES(tangents);
    ADD_BUFFER_BYTES(uv0);
    ADD_BUFFER_BYTES(vertex_colors);
    ADD_BUFFER_BYTES(triangles);
    ADD_BUFFER_BYTES(meshes);
    ADD_BUFFER_BYTES(spheres);
    ADD_BUFFER_BYTES(moving_spheres);
    ADD_BUFFER_BYTES(transforms);
    ADD_BUFFER_BYTES(instances);
    ADD_BUFFER_BYTES(material_bindings);
    ADD_BUFFER_BYTES(emitter_bindings);
    ADD_BUFFER_BYTES(aggregates);
    ADD_BUFFER_BYTES(aggregate_instance_indices);
    ADD_BUFFER_BYTES(bvh_nodes);
    ADD_BUFFER_BYTES(media);
    ADD_BUFFER_BYTES(materials);
    ADD_BUFFER_BYTES(texture_nodes);
    ADD_BUFFER_BYTES(images);
    ADD_BUFFER_BYTES(image_texels);
    ADD_BUFFER_BYTES(perlin_tables);
    ADD_BUFFER_BYTES(perlin_gradients);
    ADD_BUFFER_BYTES(perlin_permutations);
    ADD_BUFFER_BYTES(lights);
    ADD_BUFFER_BYTES(delta_light_indices);
    ADD_BUFFER_BYTES(non_delta_light_indices);
    ADD_BUFFER_BYTES(light_selection_probabilities);
    ADD_BUFFER_BYTES(light_cdf);
    ADD_BUFFER_BYTES(light_element_indices);
    ADD_BUFFER_BYTES(light_distributions);
#undef ADD_BUFFER_BYTES
    return total;
}

} // namespace cuda_backend

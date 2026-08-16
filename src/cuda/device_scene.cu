#include "device_scene.h"

#include <chrono>
#include <cstddef>
#include <cstring>
#include <sstream>
#include <vector>
#include <stdexcept>

namespace cuda_backend {
namespace {

constexpr std::size_t kSceneArenaAlignment = 16;

std::size_t aligned_offset(std::size_t offset, std::size_t alignment) {
    const std::size_t remainder = offset % alignment;
    return remainder == 0 ? offset : offset + (alignment - remainder);
}

template <typename T>
std::size_t stage_arena(std::vector<std::byte> &staging, std::size_t offset,
                        const std::vector<T> &values) {
    const std::size_t alignment =
        alignof(T) > kSceneArenaAlignment ? alignof(T)
                                          : kSceneArenaAlignment;
    offset = aligned_offset(offset, alignment);
    const std::size_t end = offset + values.size() * sizeof(T);
    if (staging.size() < end) {
        staging.resize(end);
    }
    if (!values.empty()) {
        std::memcpy(staging.data() + offset, values.data(),
                    values.size() * sizeof(T));
    }
    return end;
}

template <typename T>
std::size_t place_arena(PackedArrayView<T> &view,
                        const std::vector<T> &values, std::byte *base,
                        std::size_t offset) {
    const std::size_t alignment =
        alignof(T) > kSceneArenaAlignment ? alignof(T)
                                          : kSceneArenaAlignment;
    offset = aligned_offset(offset, alignment);
    view = {reinterpret_cast<const T *>(base + offset),
            static_cast<std::uint32_t>(values.size())};
    return offset + values.size() * sizeof(T);
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

    // Stage every buffer into one host allocation with the exact layout the
    // device arena will use.
    std::vector<std::byte> staging;
    std::size_t offset = 0;
#define STAGE_ARENA_FIELD(field)                                              \
    offset = stage_arena(staging, offset, scene.field)
    STAGE_ARENA_FIELD(positions);
    STAGE_ARENA_FIELD(normals);
    STAGE_ARENA_FIELD(tangents);
    STAGE_ARENA_FIELD(uv0);
    STAGE_ARENA_FIELD(vertex_colors);
    STAGE_ARENA_FIELD(triangles);
    STAGE_ARENA_FIELD(meshes);
    STAGE_ARENA_FIELD(spheres);
    STAGE_ARENA_FIELD(moving_spheres);
    STAGE_ARENA_FIELD(transforms);
    STAGE_ARENA_FIELD(instances);
    STAGE_ARENA_FIELD(material_bindings);
    STAGE_ARENA_FIELD(emitter_bindings);
    STAGE_ARENA_FIELD(aggregates);
    STAGE_ARENA_FIELD(aggregate_instance_indices);
    STAGE_ARENA_FIELD(bvh_nodes);
    STAGE_ARENA_FIELD(media);
    STAGE_ARENA_FIELD(materials);
    STAGE_ARENA_FIELD(texture_nodes);
    STAGE_ARENA_FIELD(images);
    STAGE_ARENA_FIELD(image_texels);
    STAGE_ARENA_FIELD(perlin_tables);
    STAGE_ARENA_FIELD(perlin_gradients);
    STAGE_ARENA_FIELD(perlin_permutations);
    STAGE_ARENA_FIELD(lights);
    STAGE_ARENA_FIELD(delta_light_indices);
    STAGE_ARENA_FIELD(non_delta_light_indices);
    STAGE_ARENA_FIELD(light_selection_probabilities);
    STAGE_ARENA_FIELD(light_cdf);
    STAGE_ARENA_FIELD(light_element_indices);
    STAGE_ARENA_FIELD(light_distributions);
#undef STAGE_ARENA_FIELD
    const std::size_t arena_bytes = staging.size();
    if (offset != arena_bytes) {
        throw std::logic_error("CUDA scene arena staging size mismatch");
    }

    staged.m_storage.upload(staging);

    CompiledSceneView &view = staged.m_view;
    view.camera = scene.camera;
    view.background = scene.background;
    view.scene_time0 = scene.scene_time0;
    view.scene_time1 = scene.scene_time1;
    std::byte *base = staged.m_storage.data();
    offset = 0;
#define PLACE_ARENA_FIELD(field)                                              \
    offset = place_arena(view.field, scene.field, base, offset)
    PLACE_ARENA_FIELD(positions);
    PLACE_ARENA_FIELD(normals);
    PLACE_ARENA_FIELD(tangents);
    PLACE_ARENA_FIELD(uv0);
    PLACE_ARENA_FIELD(vertex_colors);
    PLACE_ARENA_FIELD(triangles);
    PLACE_ARENA_FIELD(meshes);
    PLACE_ARENA_FIELD(spheres);
    PLACE_ARENA_FIELD(moving_spheres);
    PLACE_ARENA_FIELD(transforms);
    PLACE_ARENA_FIELD(instances);
    PLACE_ARENA_FIELD(material_bindings);
    PLACE_ARENA_FIELD(emitter_bindings);
    PLACE_ARENA_FIELD(aggregates);
    PLACE_ARENA_FIELD(aggregate_instance_indices);
    PLACE_ARENA_FIELD(bvh_nodes);
    PLACE_ARENA_FIELD(media);
    PLACE_ARENA_FIELD(materials);
    PLACE_ARENA_FIELD(texture_nodes);
    PLACE_ARENA_FIELD(images);
    PLACE_ARENA_FIELD(image_texels);
    PLACE_ARENA_FIELD(perlin_tables);
    PLACE_ARENA_FIELD(perlin_gradients);
    PLACE_ARENA_FIELD(perlin_permutations);
    PLACE_ARENA_FIELD(lights);
    PLACE_ARENA_FIELD(delta_light_indices);
    PLACE_ARENA_FIELD(non_delta_light_indices);
    PLACE_ARENA_FIELD(light_selection_probabilities);
    PLACE_ARENA_FIELD(light_cdf);
    PLACE_ARENA_FIELD(light_element_indices);
    PLACE_ARENA_FIELD(light_distributions);
#undef PLACE_ARENA_FIELD
    if (offset != arena_bytes) {
        throw std::logic_error("CUDA scene arena layout size mismatch");
    }

    if (staged.allocated_bytes() != arena_bytes) {
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
    result.scene = m_view;
    return result;
}

std::size_t DeviceSceneStorage::allocated_bytes() const noexcept {
    return m_storage.bytes();
}

} // namespace cuda_backend

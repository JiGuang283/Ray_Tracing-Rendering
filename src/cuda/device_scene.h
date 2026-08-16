#ifndef DEVICE_SCENE_H
#define DEVICE_SCENE_H

#include "device_buffer.h"
#include "compiled_scene.h"

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
    // One device allocation holds every scene buffer. This avoids ~30
    // cudaMalloc/cudaMemcpy pairs and keeps scene_bytes accounting exact.
    DeviceBuffer<std::byte> m_storage;
    CompiledSceneView m_view;
};

} // namespace cuda_backend

#endif

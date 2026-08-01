#ifndef DEVICE_BUFFER_H
#define DEVICE_BUFFER_H

#include "cuda_error.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace cuda_backend {

template <typename T> class DeviceBuffer {
public:
    static_assert(std::is_trivially_copyable_v<T>);

    DeviceBuffer() = default;

    ~DeviceBuffer() {
        reset();
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    DeviceBuffer(DeviceBuffer &&other) noexcept {
        swap(other);
    }

    DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
        if (this != &other) {
            reset();
            swap(other);
        }
        return *this;
    }

    void allocate(std::size_t count) {
        if (count > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("CUDA buffer exceeds uint32 capacity");
        }
        if (count == 0) {
            reset();
            return;
        }
        T *new_data = nullptr;
        RT_CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&new_data),
                                 count * sizeof(T)));
        reset();
        m_data = new_data;
        m_count = static_cast<std::uint32_t>(count);
    }

    void upload(const std::vector<T> &source) {
        DeviceBuffer uploaded;
        uploaded.allocate(source.size());
        if (!source.empty()) {
            RT_CUDA_CHECK(cudaMemcpy(uploaded.m_data, source.data(),
                                     source.size() * sizeof(T),
                                     cudaMemcpyHostToDevice));
        }
        swap(uploaded);
    }

    void download(std::vector<T> &destination) const {
        destination.resize(m_count);
        if (m_count != 0) {
            RT_CUDA_CHECK(cudaMemcpy(destination.data(), m_data, bytes(),
                                     cudaMemcpyDeviceToHost));
        }
    }

    void reset() noexcept {
        if (m_data != nullptr) {
            cudaFree(m_data);
        }
        m_data = nullptr;
        m_count = 0;
    }

    void swap(DeviceBuffer &other) noexcept {
        std::swap(m_data, other.m_data);
        std::swap(m_count, other.m_count);
    }

    T *data() noexcept {
        return m_data;
    }

    const T *data() const noexcept {
        return m_data;
    }

    std::uint32_t size() const noexcept {
        return m_count;
    }

    std::size_t bytes() const noexcept {
        return static_cast<std::size_t>(m_count) * sizeof(T);
    }

private:
    T *m_data = nullptr;
    std::uint32_t m_count = 0;
};

} // namespace cuda_backend

#endif

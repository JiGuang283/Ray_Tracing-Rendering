#ifndef CUDA_ERROR_H
#define CUDA_ERROR_H

#include <cuda_runtime_api.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace cuda_backend {

class CudaError : public std::runtime_error {
public:
    explicit CudaError(const std::string &message)
        : std::runtime_error(message) {
    }
};

inline void check_cuda(cudaError_t status, const char *expression,
                       const char *file, int line) {
    if (status == cudaSuccess) {
        return;
    }
    std::ostringstream message;
    message << expression << " failed at " << file << ':' << line << ": "
            << cudaGetErrorString(status) << " ("
            << static_cast<int>(status) << ')';
    throw CudaError(message.str());
}

} // namespace cuda_backend

#define RT_CUDA_CHECK(expression)                                             \
    ::cuda_backend::check_cuda((expression), #expression, __FILE__, __LINE__)

#endif

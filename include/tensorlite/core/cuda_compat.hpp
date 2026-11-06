#pragma once

#include <cstddef>
#include <string>
#include <vector>
#include <cstring>
#include "tensorlite/core/error.hpp"

#if defined(TENSORLITE_USE_CUDA)
#include <cuda_runtime.h>
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = (call);                                                 \
        if (err != cudaSuccess) {                                                 \
            throw ::tensorlite::TensorLiteException(                              \
                std::format("CUDA error at {}:{}: {}", __FILE__, __LINE__,        \
                            cudaGetErrorString(err)));                            \
        }                                                                         \
    } while (0)
#else
// Simulated CUDA driver / runtime for CPU environments
using cudaStream_t = void*;
using cudaError_t = int;
constexpr int cudaSuccess = 0;

inline const char* cudaGetErrorString(cudaError_t) {
    return "Simulated CUDA Success";
}

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        int err = (call);                                                         \
        if (err != 0) {                                                           \
            throw ::tensorlite::TensorLiteException(                              \
                std::format("Simulated CUDA error at {}:{}", __FILE__, __LINE__));\
        }                                                                         \
    } while (0)
#endif

namespace tensorlite {

enum class CudaMemcpyKind {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
    HostToHost
};

class CudaPlatform {
public:
    static void* allocate_device(size_t bytes);
    static void free_device(void* ptr);

    static void* allocate_pinned_host(size_t bytes);
    static void free_pinned_host(void* ptr);

    static void memcpy_sync(void* dst, const void* src, size_t bytes, CudaMemcpyKind kind);
    static void synchronize();
    static bool is_cuda_available();
};

} // namespace tensorlite

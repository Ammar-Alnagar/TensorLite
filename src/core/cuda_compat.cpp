#include "tensorlite/core/cuda_compat.hpp"
#include <cstdlib>
#include <new>

namespace tensorlite {

#if defined(TENSORLITE_USE_CUDA)

void* CudaPlatform::allocate_device(size_t bytes) {
    if (bytes == 0) return nullptr;
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        throw OutOfMemoryException(std::format("cudaMalloc failed for {} bytes: {}", bytes, cudaGetErrorString(err)));
    }
    return ptr;
}

void CudaPlatform::free_device(void* ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

void* CudaPlatform::allocate_pinned_host(size_t bytes) {
    if (bytes == 0) return nullptr;
    void* ptr = nullptr;
    cudaError_t err = cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault);
    if (err != cudaSuccess) {
        throw OutOfMemoryException(std::format("cudaHostAlloc failed for {} bytes: {}", bytes, cudaGetErrorString(err)));
    }
    return ptr;
}

void CudaPlatform::free_pinned_host(void* ptr) {
    if (ptr) {
        cudaFreeHost(ptr);
    }
}

void CudaPlatform::memcpy_sync(void* dst, const void* src, size_t bytes, CudaMemcpyKind kind) {
    if (bytes == 0 || !dst || !src) return;
    cudaMemcpyKind ckind = cudaMemcpyDefault;
    switch (kind) {
        case CudaMemcpyKind::HostToDevice: ckind = cudaMemcpyHostToDevice; break;
        case CudaMemcpyKind::DeviceToHost: ckind = cudaMemcpyDeviceToHost; break;
        case CudaMemcpyKind::DeviceToDevice: ckind = cudaMemcpyDeviceToDevice; break;
        case CudaMemcpyKind::HostToHost: ckind = cudaMemcpyHostToHost; break;
    }
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, ckind));
}

void CudaPlatform::synchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

bool CudaPlatform::is_cuda_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}

#else

// CPU Simulated memory management
void* CudaPlatform::allocate_device(size_t bytes) {
    if (bytes == 0) return nullptr;
    // 64-byte aligned allocation for AVX512/SIMD compatibility
    void* ptr = ::aligned_alloc(64, ((bytes + 63) / 64) * 64);
    if (!ptr) {
        throw OutOfMemoryException(std::format("CPU aligned_alloc failed for {} bytes", bytes));
    }
    std::memset(ptr, 0, bytes);
    return ptr;
}

void CudaPlatform::free_device(void* ptr) {
    if (ptr) {
        std::free(ptr);
    }
}

void* CudaPlatform::allocate_pinned_host(size_t bytes) {
    return allocate_device(bytes);
}

void CudaPlatform::free_pinned_host(void* ptr) {
    free_device(ptr);
}

void CudaPlatform::memcpy_sync(void* dst, const void* src, size_t bytes, CudaMemcpyKind /*kind*/) {
    if (bytes == 0 || !dst || !src) return;
    std::memcpy(dst, src, bytes);
}

void CudaPlatform::synchronize() {
    // No-op for CPU simulation
}

bool CudaPlatform::is_cuda_available() {
    return false; // CPU simulation active
}

#endif

} // namespace tensorlite

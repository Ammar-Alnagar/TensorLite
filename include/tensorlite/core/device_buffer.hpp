#pragma once

#include <cstddef>
#include <span>
#include <vector>
#include <utility>
#include "tensorlite/core/types.hpp"
#include "tensorlite/core/cuda_compat.hpp"
#include "tensorlite/core/error.hpp"

namespace tensorlite {

// Forward declarations
template <typename T>
class TensorView;

// DeviceBuffer manages raw GPU/accelerator memory via strict RAII
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(size_t size_in_bytes)
        : size_in_bytes_(size_in_bytes),
          ptr_(CudaPlatform::allocate_device(size_in_bytes)) {}

    ~DeviceBuffer() noexcept {
        release();
    }

    // Move semantics (no copying to prevent accidental device double-frees)
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : size_in_bytes_(other.size_in_bytes_), ptr_(other.ptr_) {
        other.size_in_bytes_ = 0;
        other.ptr_ = nullptr;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            size_in_bytes_ = other.size_in_bytes_;
            ptr_ = other.ptr_;
            other.size_in_bytes_ = 0;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    [[nodiscard]] void* raw_ptr() noexcept { return ptr_; }
    [[nodiscard]] const void* raw_ptr() const noexcept { return ptr_; }

    template <typename T>
    [[nodiscard]] T* as() noexcept {
        return static_cast<T*>(ptr_);
    }

    template <typename T>
    [[nodiscard]] const T* as() const noexcept {
        return static_cast<const T*>(ptr_);
    }

    [[nodiscard]] size_t size_bytes() const noexcept { return size_in_bytes_; }
    [[nodiscard]] bool empty() const noexcept { return ptr_ == nullptr || size_in_bytes_ == 0; }

    void copy_from_host(const void* host_ptr, size_t bytes) {
        if (bytes > size_in_bytes_) {
            throw InvalidArgumentException(std::format(
                "Host copy overflow: target buffer size {} bytes, copy requested {} bytes",
                size_in_bytes_, bytes));
        }
        CudaPlatform::memcpy_sync(ptr_, host_ptr, bytes, CudaMemcpyKind::HostToDevice);
    }

    void copy_to_host(void* host_ptr, size_t bytes) const {
        if (bytes > size_in_bytes_) {
            throw InvalidArgumentException(std::format(
                "Device copy overflow: target buffer size {} bytes, copy requested {} bytes",
                size_in_bytes_, bytes));
        }
        CudaPlatform::memcpy_sync(host_ptr, ptr_, bytes, CudaMemcpyKind::DeviceToHost);
    }

    void zero() {
        if (ptr_ && size_in_bytes_ > 0) {
            std::vector<uint8_t> zeros(size_in_bytes_, 0);
            copy_from_host(zeros.data(), size_in_bytes_);
        }
    }

private:
    void release() noexcept {
        if (ptr_) {
            CudaPlatform::free_device(ptr_);
            ptr_ = nullptr;
            size_in_bytes_ = 0;
        }
    }

    size_t size_in_bytes_{0};
    void* ptr_{nullptr};
};

// PinnedHostBuffer manages page-locked (pinned) CPU memory for asynchronous DMA transfers
class PinnedHostBuffer {
public:
    PinnedHostBuffer() = default;

    explicit PinnedHostBuffer(size_t size_in_bytes)
        : size_in_bytes_(size_in_bytes),
          ptr_(CudaPlatform::allocate_pinned_host(size_in_bytes)) {}

    ~PinnedHostBuffer() noexcept {
        release();
    }

    PinnedHostBuffer(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

    PinnedHostBuffer(PinnedHostBuffer&& other) noexcept
        : size_in_bytes_(other.size_in_bytes_), ptr_(other.ptr_) {
        other.size_in_bytes_ = 0;
        other.ptr_ = nullptr;
    }

    PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept {
        if (this != &other) {
            release();
            size_in_bytes_ = other.size_in_bytes_;
            ptr_ = other.ptr_;
            other.size_in_bytes_ = 0;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    [[nodiscard]] void* raw_ptr() noexcept { return ptr_; }
    [[nodiscard]] const void* raw_ptr() const noexcept { return ptr_; }

    template <typename T>
    [[nodiscard]] T* as() noexcept {
        return static_cast<T*>(ptr_);
    }

    template <typename T>
    [[nodiscard]] const T* as() const noexcept {
        return static_cast<const T*>(ptr_);
    }

    [[nodiscard]] size_t size_bytes() const noexcept { return size_in_bytes_; }

private:
    void release() noexcept {
        if (ptr_) {
            CudaPlatform::free_pinned_host(ptr_);
            ptr_ = nullptr;
            size_in_bytes_ = 0;
        }
    }

    size_t size_in_bytes_{0};
    void* ptr_{nullptr};
};

} // namespace tensorlite

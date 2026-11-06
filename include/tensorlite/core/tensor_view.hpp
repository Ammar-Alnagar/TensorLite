#pragma once

#include <vector>
#include <numeric>
#include <cassert>
#include <span>
#include <initializer_list>
#include "tensorlite/core/types.hpp"
#include "tensorlite/core/device_buffer.hpp"

namespace tensorlite {

// Non-owning multi-dimensional view over contiguous or strided memory
template <typename T>
class TensorView {
public:
    TensorView() : data_(nullptr) {}

    TensorView(T* data, std::vector<int64_t> shape)
        : data_(data), shape_(std::move(shape)) {
        compute_contiguous_strides();
    }

    TensorView(T* data, std::vector<int64_t> shape, std::vector<int64_t> strides)
        : data_(data), shape_(std::move(shape)), strides_(std::move(strides)) {}

    [[nodiscard]] T* data() noexcept { return data_; }
    [[nodiscard]] const T* data() const noexcept { return data_; }

    [[nodiscard]] const std::vector<int64_t>& shape() const noexcept { return shape_; }
    [[nodiscard]] const std::vector<int64_t>& strides() const noexcept { return strides_; }

    [[nodiscard]] int64_t dim(size_t index) const {
        if (index >= shape_.size()) {
            throw InvalidArgumentException("Dimension index out of bounds");
        }
        return shape_[index];
    }

    [[nodiscard]] size_t ndim() const noexcept { return shape_.size(); }

    [[nodiscard]] int64_t numel() const noexcept {
        if (shape_.empty()) return 0;
        return std::accumulate(shape_.begin(), shape_.end(), int64_t{1}, std::multiplies<int64_t>());
    }

    [[nodiscard]] size_t size_bytes() const noexcept {
        return numel() * sizeof(T);
    }

    // 1D access
    [[nodiscard]] T& operator()(int64_t i) {
        return data_[i * strides_[0]];
    }
    [[nodiscard]] const T& operator()(int64_t i) const {
        return data_[i * strides_[0]];
    }

    // 2D access
    [[nodiscard]] T& operator()(int64_t i, int64_t j) {
        return data_[i * strides_[0] + j * strides_[1]];
    }
    [[nodiscard]] const T& operator()(int64_t i, int64_t j) const {
        return data_[i * strides_[0] + j * strides_[1]];
    }

    // 3D access
    [[nodiscard]] T& operator()(int64_t i, int64_t j, int64_t k) {
        return data_[i * strides_[0] + j * strides_[1] + k * strides_[2]];
    }
    [[nodiscard]] const T& operator()(int64_t i, int64_t j, int64_t k) const {
        return data_[i * strides_[0] + j * strides_[1] + k * strides_[2]];
    }

    // 4D access
    [[nodiscard]] T& operator()(int64_t i, int64_t j, int64_t k, int64_t l) {
        return data_[i * strides_[0] + j * strides_[1] + k * strides_[2] + l * strides_[3]];
    }
    [[nodiscard]] const T& operator()(int64_t i, int64_t j, int64_t k, int64_t l) const {
        return data_[i * strides_[0] + j * strides_[1] + k * strides_[2] + l * strides_[3]];
    }

    // Slice along the leading dimension
    [[nodiscard]] TensorView<T> slice(int64_t start, int64_t length) const {
        if (shape_.empty()) {
            throw InvalidArgumentException("Cannot slice empty TensorView");
        }
        if (start < 0 || start + length > shape_[0]) {
            throw InvalidArgumentException("Slice bounds exceeded");
        }
        std::vector<int64_t> new_shape = shape_;
        new_shape[0] = length;
        T* new_data = data_ + start * strides_[0];
        return TensorView<T>(new_data, std::move(new_shape), strides_);
    }

    // Convert contiguous view to std::span
    [[nodiscard]] std::span<T> as_span() {
        return std::span<T>(data_, static_cast<size_t>(numel()));
    }
    [[nodiscard]] std::span<const T> as_span() const {
        return std::span<const T>(data_, static_cast<size_t>(numel()));
    }

private:
    void compute_contiguous_strides() {
        strides_.resize(shape_.size());
        if (shape_.empty()) return;
        int64_t stride = 1;
        for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
            strides_[i] = stride;
            stride *= shape_[i];
        }
    }

    T* data_{nullptr};
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
};

// Strongly typed Tensor owning a DeviceBuffer
template <typename T>
class DeviceTensor {
public:
    DeviceTensor() = default;

    explicit DeviceTensor(std::vector<int64_t> shape)
        : shape_(shape) {
        int64_t count = 1;
        for (auto d : shape_) count *= d;
        buffer_ = DeviceBuffer(count * sizeof(T));
    }

    [[nodiscard]] TensorView<T> view() {
        return TensorView<T>(buffer_.template as<T>(), shape_);
    }

    [[nodiscard]] const TensorView<const T> view() const {
        return TensorView<const T>(buffer_.template as<const T>(), shape_);
    }

    [[nodiscard]] DeviceBuffer& buffer() noexcept { return buffer_; }
    [[nodiscard]] const DeviceBuffer& buffer() const noexcept { return buffer_; }
    [[nodiscard]] const std::vector<int64_t>& shape() const noexcept { return shape_; }

private:
    std::vector<int64_t> shape_;
    DeviceBuffer buffer_;
};

} // namespace tensorlite

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <memory>
#include <span>
#include "tensorlite/core/types.hpp"
#include "tensorlite/core/device_buffer.hpp"
#include "tensorlite/core/error.hpp"

namespace tensorlite {

// Metadata describing an individual tensor inside safetensors
struct TensorMeta {
    std::string name;
    DataType dtype{DataType::FP32};
    std::vector<int64_t> shape;
    size_t data_offset_start{0};
    size_t data_offset_end{0};

    [[nodiscard]] size_t size_bytes() const noexcept {
        return data_offset_end - data_offset_start;
    }
};

// Represents a loaded tensor on device, holding its FP8 or FP32 weights + optional inverse scale
struct WeightTensor {
    std::string name;
    DataType dtype{DataType::FP8_E4M3};
    std::vector<int64_t> shape;
    DeviceBuffer data;
    float scale{1.0f}; // Dequantization scale: w_fp32 = w_fp8 * scale
};

// High-performance parser and reader for Hugging Face `.safetensors` files
class SafetensorsLoader {
public:
    explicit SafetensorsLoader(const std::filesystem::path& file_path);
    ~SafetensorsLoader();

    SafetensorsLoader(const SafetensorsLoader&) = delete;
    SafetensorsLoader& operator=(const SafetensorsLoader&) = delete;
    SafetensorsLoader(SafetensorsLoader&&) noexcept;
    SafetensorsLoader& operator=(SafetensorsLoader&&) noexcept;

    [[nodiscard]] const std::unordered_map<std::string, TensorMeta>& metadata() const noexcept {
        return tensors_;
    }

    [[nodiscard]] bool contains(const std::string& tensor_name) const {
        return tensors_.find(tensor_name) != tensors_.end();
    }

    // Load tensor data directly to DeviceBuffer
    [[nodiscard]] WeightTensor load_tensor(const std::string& name);

    // Read all weights in file and map them into the weight map
    void load_all(std::unordered_map<std::string, WeightTensor>& out_weights);

private:
    void parse_header();

    std::filesystem::path file_path_;
    int file_descriptor_{-1};
    void* mapped_data_{nullptr};
    size_t file_size_{0};
    size_t header_size_{0};
    std::unordered_map<std::string, TensorMeta> tensors_;
};

} // namespace tensorlite

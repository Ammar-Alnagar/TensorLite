#include "tensorlite/model/safetensors_loader.hpp"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <nlohmann/json.hpp>
#include "tensorlite/core/error.hpp"

namespace tensorlite {

static DataType parse_safetensors_dtype(const std::string& str) {
    if (str == "F32") return DataType::FP32;
    if (str == "F16") return DataType::FP16;
    if (str == "BF16") return DataType::BF16;
    if (str == "F8_E4M3" || str == "FP8_E4M3") return DataType::FP8_E4M3;
    if (str == "I32") return DataType::INT32;
    if (str == "I64") return DataType::INT64;
    if (str == "U8") return DataType::UINT8;
    return DataType::FP32;
}

SafetensorsLoader::SafetensorsLoader(const std::filesystem::path& file_path)
    : file_path_(file_path) {
    if (!std::filesystem::exists(file_path)) {
        throw ModelLoadException(std::format("Safetensors file not found: {}", file_path.string()));
    }

    file_size_ = std::filesystem::file_size(file_path);
    if (file_size_ < 8) {
        throw ModelLoadException(std::format("File too small to be valid safetensors: {}", file_path.string()));
    }

    file_descriptor_ = ::open(file_path.c_str(), O_RDONLY);
    if (file_descriptor_ < 0) {
        throw ModelLoadException(std::format("Failed to open file: {}", file_path.string()));
    }

    mapped_data_ = ::mmap(nullptr, file_size_, PROT_READ, MAP_SHARED, file_descriptor_, 0);
    if (mapped_data_ == MAP_FAILED) {
        ::close(file_descriptor_);
        file_descriptor_ = -1;
        mapped_data_ = nullptr;
        throw ModelLoadException(std::format("mmap failed for: {}", file_path.string()));
    }

    parse_header();
}

SafetensorsLoader::~SafetensorsLoader() {
    if (mapped_data_ && mapped_data_ != MAP_FAILED) {
        ::munmap(mapped_data_, file_size_);
    }
    if (file_descriptor_ >= 0) {
        ::close(file_descriptor_);
    }
}

SafetensorsLoader::SafetensorsLoader(SafetensorsLoader&& other) noexcept
    : file_path_(std::move(other.file_path_)),
      file_descriptor_(other.file_descriptor_),
      mapped_data_(other.mapped_data_),
      file_size_(other.file_size_),
      header_size_(other.header_size_),
      tensors_(std::move(other.tensors_)) {
    other.file_descriptor_ = -1;
    other.mapped_data_ = nullptr;
    other.file_size_ = 0;
    other.header_size_ = 0;
}

SafetensorsLoader& SafetensorsLoader::operator=(SafetensorsLoader&& other) noexcept {
    if (this != &other) {
        if (mapped_data_ && mapped_data_ != MAP_FAILED) {
            ::munmap(mapped_data_, file_size_);
        }
        if (file_descriptor_ >= 0) {
            ::close(file_descriptor_);
        }
        file_path_ = std::move(other.file_path_);
        file_descriptor_ = other.file_descriptor_;
        mapped_data_ = other.mapped_data_;
        file_size_ = other.file_size_;
        header_size_ = other.header_size_;
        tensors_ = std::move(other.tensors_);

        other.file_descriptor_ = -1;
        other.mapped_data_ = nullptr;
        other.file_size_ = 0;
        other.header_size_ = 0;
    }
    return *this;
}

void SafetensorsLoader::parse_header() {
    // First 8 bytes: unsigned 64-bit little-endian integer N (header length in bytes)
    const uint8_t* raw = static_cast<const uint8_t*>(mapped_data_);
    uint64_t n = 0;
    std::memcpy(&n, raw, sizeof(uint64_t));

    header_size_ = n;
    if (8 + header_size_ > file_size_) {
        throw ModelLoadException("Safetensors header size overflows file length");
    }

    std::string_view header_str(reinterpret_cast<const char*>(raw + 8), header_size_);
    nlohmann::json header_json;
    try {
        header_json = nlohmann::json::parse(header_str);
    } catch (const std::exception& e) {
        throw ModelLoadException(std::format("Safetensors JSON header parse error: {}", e.what()));
    }

    for (auto it = header_json.begin(); it != header_json.end(); ++it) {
        if (it.key() == "__metadata__") {
            continue;
        }

        const auto& tensor_info = it.value();
        TensorMeta meta;
        meta.name = it.key();
        meta.dtype = parse_safetensors_dtype(tensor_info.value("dtype", "F32"));
        meta.shape = tensor_info.value("shape", std::vector<int64_t>{});

        auto offsets = tensor_info.value("data_offsets", std::vector<size_t>{});
        if (offsets.size() != 2) {
            throw ModelLoadException(std::format("Malformed data_offsets for tensor {}", meta.name));
        }
        meta.data_offset_start = offsets[0];
        meta.data_offset_end = offsets[1];

        tensors_[meta.name] = meta;
    }
}

WeightTensor SafetensorsLoader::load_tensor(const std::string& name) {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw ModelLoadException(std::format("Tensor '{}' not found in safetensors", name));
    }

    const auto& meta = it->second;
    size_t payload_base = 8 + header_size_;
    size_t abs_start = payload_base + meta.data_offset_start;
    size_t abs_end = payload_base + meta.data_offset_end;

    if (abs_end > file_size_) {
        throw ModelLoadException(std::format("Tensor '{}' data bounds exceed file size", name));
    }

    size_t bytes = meta.size_bytes();
    const uint8_t* src_ptr = static_cast<const uint8_t*>(mapped_data_) + abs_start;

    WeightTensor wt;
    wt.name = meta.name;
    wt.dtype = meta.dtype;
    wt.shape = meta.shape;
    wt.data = DeviceBuffer(bytes);
    wt.data.copy_from_host(src_ptr, bytes);
    wt.scale = 1.0f;

    // Check if an associated scale tensor exists (e.g., name + ".weight_scale")
    std::string scale_name = name + "_scale";
    auto sit = tensors_.find(scale_name);
    if (sit != tensors_.end() && sit->second.size_bytes() == sizeof(float)) {
        size_t s_abs = payload_base + sit->second.data_offset_start;
        float s_val = 1.0f;
        std::memcpy(&s_val, static_cast<const uint8_t*>(mapped_data_) + s_abs, sizeof(float));
        wt.scale = s_val;
    }

    return wt;
}

void SafetensorsLoader::load_all(std::unordered_map<std::string, WeightTensor>& out_weights) {
    for (const auto& [name, meta] : tensors_) {
        // Skip separate scale tensors from main weight keys, as they get attached
        if (name.ends_with("_scale")) continue;
        out_weights[name] = load_tensor(name);
    }
}

} // namespace tensorlite

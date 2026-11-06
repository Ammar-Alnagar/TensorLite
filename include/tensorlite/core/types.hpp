#pragma once

#include <cstdint>
#include <string>
#include <concepts>
#include <format>
#include <functional>

namespace tensorlite {

// Strong type wrapper using CRTP-free minimal modern C++ struct idiom
template <typename Tag, typename T = int64_t>
struct StrongType {
    T value{};

    constexpr StrongType() = default;
    constexpr explicit StrongType(T val) : value(val) {}

    [[nodiscard]] constexpr T get() const noexcept { return value; }

    constexpr auto operator<=>(const StrongType&) const = default;
    constexpr bool operator==(const StrongType&) const = default;

    constexpr explicit operator T() const noexcept { return value; }
};

// Domain strong identifiers
struct SequenceIdTag {};
using SequenceId = StrongType<SequenceIdTag, int64_t>;

struct RequestIdTag {};
using RequestId = StrongType<RequestIdTag, std::string>;

struct BlockIdTag {};
using BlockId = StrongType<BlockIdTag, int32_t>;

struct TokenIdTag {};
using TokenId = StrongType<TokenIdTag, int32_t>;

inline constexpr BlockId INVALID_BLOCK_ID{-1};
inline constexpr TokenId INVALID_TOKEN_ID{-1};

// Floating point 8 (FP8 E4M3) representation and bitwise conversions
// Follows IEEE/OCP FP8 E4M3 (1 sign bit, 4 exponent bits, 3 mantissa bits, bias 7)
// Exponent = 0b1111 (15), Mantissa = 0b111 (7) -> NaN. Values up to 448.0f are finite.
struct alignas(1) fp8_e4m3 {
    uint8_t bits{0};

    constexpr fp8_e4m3() = default;
    constexpr explicit fp8_e4m3(uint8_t raw) : bits(raw) {}

    // Convert 32-bit float to FP8 E4M3
    static fp8_e4m3 from_float(float f);

    // Convert FP8 E4M3 to 32-bit float
    [[nodiscard]] float to_float() const;

    constexpr bool operator==(const fp8_e4m3& other) const = default;
};

// Data type enumeration for Tensor buffers
enum class DataType : uint8_t {
    FP32 = 0,
    FP16 = 1,
    BF16 = 2,
    FP8_E4M3 = 3,
    INT32 = 4,
    INT64 = 5,
    UINT8 = 6
};

[[nodiscard]] constexpr size_t data_type_size(DataType dt) noexcept {
    switch (dt) {
        case DataType::FP32:
        case DataType::INT32:
            return 4;
        case DataType::FP16:
        case DataType::BF16:
            return 2;
        case DataType::FP8_E4M3:
        case DataType::UINT8:
            return 1;
        case DataType::INT64:
            return 8;
    }
    return 1;
}

[[nodiscard]] constexpr const char* data_type_to_string(DataType dt) noexcept {
    switch (dt) {
        case DataType::FP32: return "FP32";
        case DataType::FP16: return "FP16";
        case DataType::BF16: return "BF16";
        case DataType::FP8_E4M3: return "FP8_E4M3";
        case DataType::INT32: return "INT32";
        case DataType::INT64: return "INT64";
        case DataType::UINT8: return "UINT8";
    }
    return "UNKNOWN";
}

} // namespace tensorlite

// Hash specializations for strong types
template <typename Tag, typename T>
struct std::hash<tensorlite::StrongType<Tag, T>> {
    std::size_t operator()(const tensorlite::StrongType<Tag, T>& id) const noexcept {
        return std::hash<T>{}(id.value);
    }
};

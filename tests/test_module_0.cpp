#include <cassert>
#include <iostream>
#include <cmath>
#include "tensorlite/core/types.hpp"
#include "tensorlite/core/error.hpp"
#include "tensorlite/core/cuda_compat.hpp"

using namespace tensorlite;

void test_strong_types() {
    SequenceId s1{42};
    SequenceId s2{42};
    SequenceId s3{100};

    assert(s1 == s2);
    assert(s1 != s3);
    assert(s1.get() == 42);

    BlockId b1{10};
    assert(b1.get() == 10);
    assert(INVALID_BLOCK_ID.get() == -1);

    std::hash<SequenceId> hasher;
    assert(hasher(s1) == hasher(s2));
    std::cout << "[PASS] Strong types verified.\n";
}

void test_fp8_conversion() {
    // Test 0.0f
    fp8_e4m3 zero = fp8_e4m3::from_float(0.0f);
    assert(zero.to_float() == 0.0f);

    // Test 1.0f
    fp8_e4m3 one = fp8_e4m3::from_float(1.0f);
    assert(std::fabs(one.to_float() - 1.0f) < 1e-3f);

    // Test -1.0f
    fp8_e4m3 neg_one = fp8_e4m3::from_float(-1.0f);
    assert(std::fabs(neg_one.to_float() - (-1.0f)) < 1e-3f);

    // Test 2.0f, 0.5f, 15.0f
    fp8_e4m3 two = fp8_e4m3::from_float(2.0f);
    assert(std::fabs(two.to_float() - 2.0f) < 1e-3f);

    fp8_e4m3 half = fp8_e4m3::from_float(0.5f);
    assert(std::fabs(half.to_float() - 0.5f) < 1e-3f);

    fp8_e4m3 val15 = fp8_e4m3::from_float(15.0f);
    assert(std::fabs(val15.to_float() - 15.0f) < 1.0f); // FP8 has 3 bits mantissa

    // Test max finite (~448.0)
    fp8_e4m3 max_val = fp8_e4m3::from_float(500.0f);
    assert(max_val.to_float() <= 448.0f);

    std::cout << "[PASS] FP8 E4M3 conversion verified.\n";
}

void test_cuda_compat_memory() {
    constexpr size_t size = 1024;
    void* dev_ptr = CudaPlatform::allocate_device(size);
    assert(dev_ptr != nullptr);

    std::vector<uint8_t> host_src(size, 0xAB);
    std::vector<uint8_t> host_dst(size, 0x00);

    CudaPlatform::memcpy_sync(dev_ptr, host_src.data(), size, CudaMemcpyKind::HostToDevice);
    CudaPlatform::memcpy_sync(host_dst.data(), dev_ptr, size, CudaMemcpyKind::DeviceToHost);

    assert(host_src == host_dst);
    CudaPlatform::free_device(dev_ptr);

    std::cout << "[PASS] CudaPlatform memory allocation & memcpy verified.\n";
}

int main() {
    std::cout << "Running Module 0 Tests...\n";
    test_strong_types();
    test_fp8_conversion();
    test_cuda_compat_memory();
    std::cout << "All Module 0 tests completed successfully!\n";
    return 0;
}

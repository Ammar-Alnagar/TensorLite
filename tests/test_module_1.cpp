#include <cassert>
#include <iostream>
#include <vector>
#include "tensorlite/core/device_buffer.hpp"
#include "tensorlite/core/tensor_view.hpp"

using namespace tensorlite;

void test_device_buffer_raii() {
    constexpr size_t size = 2048;
    DeviceBuffer buf1(size);
    assert(buf1.raw_ptr() != nullptr);
    assert(buf1.size_bytes() == size);

    // Test move construction
    DeviceBuffer buf2(std::move(buf1));
    assert(buf1.raw_ptr() == nullptr);
    assert(buf1.size_bytes() == 0);
    assert(buf2.raw_ptr() != nullptr);
    assert(buf2.size_bytes() == size);

    // Test move assignment
    DeviceBuffer buf3;
    buf3 = std::move(buf2);
    assert(buf2.raw_ptr() == nullptr);
    assert(buf3.raw_ptr() != nullptr);

    std::cout << "[PASS] DeviceBuffer RAII and move semantics verified.\n";
}

void test_tensor_view_nd() {
    // Allocate 2x3x4 float tensor = 24 floats
    std::vector<float> host_data(24);
    for (size_t i = 0; i < host_data.size(); ++i) {
        host_data[i] = static_cast<float>(i);
    }

    TensorView<float> view(host_data.data(), {2, 3, 4});
    assert(view.ndim() == 3);
    assert(view.numel() == 24);
    assert(view.dim(0) == 2);
    assert(view.dim(1) == 3);
    assert(view.dim(2) == 4);

    // Verify strides for contiguous C-layout: [12, 4, 1]
    assert(view.strides()[0] == 12);
    assert(view.strides()[1] == 4);
    assert(view.strides()[2] == 1);

    // Check indexing
    // (1, 2, 3) -> 1*12 + 2*4 + 3*1 = 12 + 8 + 3 = 23
    assert(view(1, 2, 3) == 23.0f);
    assert(view(0, 0, 0) == 0.0f);
    assert(view(0, 1, 2) == (0*12 + 1*4 + 2) );

    // Test slicing
    auto sliced = view.slice(1, 1); // 1x3x4 starting at index 1
    assert(sliced.shape()[0] == 1);
    assert(sliced.shape()[1] == 3);
    assert(sliced.shape()[2] == 4);
    assert(sliced(0, 0, 0) == 12.0f);

    // Test std::span integration
    auto sp = view.as_span();
    assert(sp.size() == 24);
    assert(sp[10] == 10.0f);

    std::cout << "[PASS] TensorView N-D indexing and slicing verified.\n";
}

void test_device_tensor() {
    DeviceTensor<float> tensor({4, 16});
    assert(tensor.shape()[0] == 4);
    assert(tensor.shape()[1] == 16);
    assert(tensor.buffer().size_bytes() == 4 * 16 * sizeof(float));

    auto v = tensor.view();
    assert(v.numel() == 64);
    v(0, 0) = 42.0f;
    assert(v(0, 0) == 42.0f);

    std::cout << "[PASS] DeviceTensor lifecycle verified.\n";
}

int main() {
    std::cout << "Running Module 1 Tests...\n";
    test_device_buffer_raii();
    test_tensor_view_nd();
    test_device_tensor();
    std::cout << "All Module 1 tests completed successfully!\n";
    return 0;
}

#include <cassert>
#include <iostream>
#include <vector>
#include <cmath>
#include "tensorlite/kernel/attention.hpp"
#include "tensorlite/core/types.hpp"

using namespace tensorlite;

void test_paged_attention_fp8_decode() {
    constexpr int32_t num_seqs = 2;
    constexpr int32_t num_heads = 4;
    constexpr int32_t num_kv_heads = 2; // GQA (2 query heads per KV head)
    constexpr int32_t head_dim = 16;
    constexpr int32_t block_size = 16;
    constexpr int32_t max_blocks = 4;

    PagedAttentionParams params;
    params.num_seqs = num_seqs;
    params.num_heads = num_heads;
    params.num_kv_heads = num_kv_heads;
    params.head_dim = head_dim;
    params.block_size = block_size;
    params.scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    params.kv_cache_scale = 1.0f;

    // Sequence lengths: Seq 0 has 20 tokens (2 blocks), Seq 1 has 5 tokens (1 block)
    std::vector<int32_t> seq_lens = {20, 5};
    // Block tables: Seq 0 uses blocks [3, 7], Seq 1 uses block [2]
    std::vector<int32_t> block_tables = {
        3, 7, -1, -1,
        2, -1, -1, -1
    };

    // Query vectors: [num_seqs, num_heads, head_dim]
    std::vector<float> q(num_seqs * num_heads * head_dim, 0.5f);
    std::vector<float> out(num_seqs * num_heads * head_dim, 0.0f);

    // Physical KV Cache in FP8: 10 blocks allocated
    size_t total_elements = 10 * block_size * num_kv_heads * head_dim;
    std::vector<fp8_e4m3> k_cache(total_elements, fp8_e4m3::from_float(0.2f));
    std::vector<fp8_e4m3> v_cache(total_elements, fp8_e4m3::from_float(1.5f));

    AttentionKernel::paged_attention_decode(
        out.data(),
        q.data(),
        k_cache.data(),
        v_cache.data(),
        block_tables.data(),
        seq_lens.data(),
        max_blocks,
        params,
        true /* is_fp8 */
    );

    // Verify output: Since all V elements are ~1.5, Softmax weights sum to 1.0,
    // so every output dimension should approximate 1.5!
    for (float val : out) {
        assert(std::fabs(val - 1.5f) < 0.1f);
    }

    std::cout << "[PASS] PagedAttention FP8 on-the-fly dequantization and GQA decode verified.\n";
}

void test_causal_attention_prefill() {
    constexpr int32_t seq_len = 4;
    constexpr int32_t num_heads = 2;
    constexpr int32_t num_kv_heads = 2;
    constexpr int32_t head_dim = 8;
    constexpr float scale = 0.3535f;

    std::vector<float> q(seq_len * num_heads * head_dim, 0.1f);
    std::vector<float> k(seq_len * num_kv_heads * head_dim, 0.1f);
    std::vector<float> v(seq_len * num_kv_heads * head_dim, 2.0f);
    std::vector<float> out(seq_len * num_heads * head_dim, 0.0f);

    AttentionKernel::causal_attention_prefill(
        out.data(), q.data(), k.data(), v.data(),
        seq_len, num_heads, num_kv_heads, head_dim, scale
    );

    for (float val : out) {
        assert(std::fabs(val - 2.0f) < 1e-3f);
    }
    std::cout << "[PASS] Causal attention prefill verified.\n";
}

int main() {
    std::cout << "Running Module 5 Tests...\n";
    test_paged_attention_fp8_decode();
    test_causal_attention_prefill();
    std::cout << "All Module 5 tests completed successfully!\n";
    return 0;
}

#pragma once

#include <span>
#include <vector>
#include <cstdint>
#include "tensorlite/core/types.hpp"
#include "tensorlite/core/device_buffer.hpp"
#include "tensorlite/cache/kv_cache_manager.hpp"

namespace tensorlite {

// Attention kernel execution context and parameters
struct PagedAttentionParams {
    int32_t num_seqs{0};             // Batch size (number of sequences)
    int32_t num_heads{0};            // Query heads (e.g. 14 or 32)
    int32_t num_kv_heads{0};         // Key/Value heads for GQA (e.g. 2 or 8)
    int32_t head_dim{0};             // Dimension per head (e.g. 64 or 128)
    int32_t block_size{16};          // Block size in tokens
    int32_t max_context_len{0};      // Maximum context length in this batch
    float scale{1.0f};               // 1.0f / sqrt(head_dim)
    float kv_cache_scale{1.0f};      // FP8 dequantization scale for KV cache
};

// Pure C++ reference kernel + CUDA kernel dispatch interface
class AttentionKernel {
public:
    // PagedAttention for Decode stage (each sequence queries its previous KV history)
    // Q: [num_seqs, num_heads, head_dim] (FP32)
    // Out: [num_seqs, num_heads, head_dim] (FP32)
    // block_tables: [num_seqs, max_blocks_per_seq] (int32_t BlockId)
    // seq_lens: [num_seqs] (int32_t)
    // k_cache / v_cache: pointers into KVCacheManager unified pool (FP8_E4M3 or FP32)
    static void paged_attention_decode(
        float* out,
        const float* q,
        const void* k_cache,
        const void* v_cache,
        const int32_t* block_tables,
        const int32_t* seq_lens,
        int32_t max_blocks_per_seq,
        const PagedAttentionParams& params,
        bool is_fp8 = true
    );

    // Standard Multi-Head / Grouped-Query Attention for Prefill stage
    // Computes causal attention for prompt tokens
    static void causal_attention_prefill(
        float* out,
        const float* q,
        const float* k,
        const float* v,
        int32_t seq_len,
        int32_t num_heads,
        int32_t num_kv_heads,
        int32_t head_dim,
        float scale
    );
};

} // namespace tensorlite

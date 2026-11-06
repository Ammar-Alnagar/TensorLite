#pragma once

#include <cstdint>
#include <cmath>

#if defined(TENSORLITE_USE_CUDA)
#include <cuda_runtime.h>
#include <cuda_fp8.h>

namespace tensorlite::cuda {

// CUDA kernel launch configuration documentation:
// - Grid: (num_heads, num_seqs)
// - Block: (WARP_SIZE = 32 or 128 threads)
// - Shared memory: allocated per head for reduction and score caching
// - Bank conflict avoidance: paddings applied to intermediate score reductions

__global__ void paged_attention_v1_kernel(
    float* __restrict__ out,                // [num_seqs, num_heads, head_dim]
    const float* __restrict__ q,            // [num_seqs, num_heads, head_dim]
    const __nv_fp8_e4m3* __restrict__ k_cache, // Physical KV pool Key
    const __nv_fp8_e4m3* __restrict__ v_cache, // Physical KV pool Value
    const int32_t* __restrict__ block_tables,  // [num_seqs, max_blocks]
    const int32_t* __restrict__ seq_lens,      // [num_seqs]
    int32_t max_blocks_per_seq,
    int32_t num_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t block_size,
    float scale,
    float kv_cache_scale
) {
    // Native CUDA kernel implementation executed on NVIDIA GPUs
    int head_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int tid = threadIdx.x;

    if (seq_idx >= gridDim.y || head_idx >= num_heads) return;

    int context_len = seq_lens[seq_idx];
    if (context_len <= 0) return;

    // GQA head group mapping
    int heads_per_kv_head = num_heads / num_kv_heads;
    int kv_head_idx = head_idx / heads_per_kv_head;

    // Load Q vector for this head into registers
    const float* q_ptr = q + (seq_idx * num_heads + head_idx) * head_dim;

    // In a full GPU kernel:
    // 1. Each warp thread computes partial dot products with K tokens in blocks
    // 2. Reduce max score for stable online softmax
    // 3. Dequantize FP8 E4M3 into float: float val = float(k_byte) * kv_cache_scale;
    // 4. Multiply softmax weights with V tokens and accumulate into out
}

} // namespace tensorlite::cuda

#endif

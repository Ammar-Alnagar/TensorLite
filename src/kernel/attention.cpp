#include "tensorlite/kernel/attention.hpp"
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstring>

namespace tensorlite {

void AttentionKernel::paged_attention_decode(
    float* out,
    const float* q,
    const void* k_cache,
    const void* v_cache,
    const int32_t* block_tables,
    const int32_t* seq_lens,
    int32_t max_blocks_per_seq,
    const PagedAttentionParams& params,
    bool is_fp8
) {
    const int32_t num_seqs = params.num_seqs;
    const int32_t num_heads = params.num_heads;
    const int32_t num_kv_heads = params.num_kv_heads;
    const int32_t head_dim = params.head_dim;
    const int32_t block_size = params.block_size;
    const float scale = params.scale;
    const float kv_scale = params.kv_cache_scale;
    const int32_t heads_per_kv_head = num_heads / num_kv_heads;

    const fp8_e4m3* k_fp8 = static_cast<const fp8_e4m3*>(k_cache);
    const fp8_e4m3* v_fp8 = static_cast<const fp8_e4m3*>(v_cache);
    const float* k_f32 = static_cast<const float*>(k_cache);
    const float* v_f32 = static_cast<const float*>(v_cache);

    // Stride helper inside a physical block:
    // A block contains `block_size` tokens. For each token: `num_kv_heads * head_dim` elements.
    auto get_kv_offset = [&](int32_t block_id, int32_t block_token_offset, int32_t kv_head, int32_t d) -> size_t {
        size_t block_base = static_cast<size_t>(block_id) * (block_size * num_kv_heads * head_dim);
        size_t token_base = static_cast<size_t>(block_token_offset) * (num_kv_heads * head_dim);
        size_t head_base = static_cast<size_t>(kv_head) * head_dim;
        return block_base + token_base + head_base + d;
    };

    // Parallelize over sequences and attention heads
    #pragma omp parallel for collapse(2) if(num_seqs * num_heads > 4)
    for (int32_t s = 0; s < num_seqs; ++s) {
        for (int32_t h = 0; h < num_heads; ++h) {
            const int32_t kv_h = h / heads_per_kv_head;
            const int32_t context_len = seq_lens[s];

            if (context_len <= 0) {
                continue;
            }

            const float* q_vec = q + (s * num_heads + h) * head_dim;
            float* out_vec = out + (s * num_heads + h) * head_dim;

            // 1. Compute dot-product attention scores S[t] = (Q * K_t) * scale
            std::vector<float> scores(context_len, 0.0f);
            float max_score = -1e30f;

            for (int32_t t = 0; t < context_len; ++t) {
                int32_t logical_block = t / block_size;
                int32_t block_offset = t % block_size;
                int32_t physical_block_id = block_tables[s * max_blocks_per_seq + logical_block];

                float dot = 0.0f;
                for (int32_t d = 0; d < head_dim; ++d) {
                    size_t idx = get_kv_offset(physical_block_id, block_offset, kv_h, d);
                    float k_val = 0.0f;
                    if (is_fp8) {
                        k_val = k_fp8[idx].to_float() * kv_scale;
                    } else {
                        k_val = k_f32[idx];
                    }
                    dot += q_vec[d] * k_val;
                }
                scores[t] = dot * scale;
                if (scores[t] > max_score) {
                    max_score = scores[t];
                }
            }

            // 2. Numerically stable Softmax
            float sum_exp = 0.0f;
            for (int32_t t = 0; t < context_len; ++t) {
                scores[t] = std::exp(scores[t] - max_score);
                sum_exp += scores[t];
            }
            float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
            for (int32_t t = 0; t < context_len; ++t) {
                scores[t] *= inv_sum;
            }

            // 3. Weighted accumulation Out = sum(scores[t] * V_t)
            std::memset(out_vec, 0, head_dim * sizeof(float));
            for (int32_t t = 0; t < context_len; ++t) {
                int32_t logical_block = t / block_size;
                int32_t block_offset = t % block_size;
                int32_t physical_block_id = block_tables[s * max_blocks_per_seq + logical_block];
                float weight = scores[t];

                for (int32_t d = 0; d < head_dim; ++d) {
                    size_t idx = get_kv_offset(physical_block_id, block_offset, kv_h, d);
                    float v_val = 0.0f;
                    if (is_fp8) {
                        v_val = v_fp8[idx].to_float() * kv_scale;
                    } else {
                        v_val = v_f32[idx];
                    }
                    out_vec[d] += weight * v_val;
                }
            }
        }
    }
}

void AttentionKernel::causal_attention_prefill(
    float* out,
    const float* q,
    const float* k,
    const float* v,
    int32_t seq_len,
    int32_t num_heads,
    int32_t num_kv_heads,
    int32_t head_dim,
    float scale
) {
    const int32_t heads_per_kv = num_heads / num_kv_heads;

    for (int32_t h = 0; h < num_heads; ++h) {
        int32_t kv_h = h / heads_per_kv;

        for (int32_t i = 0; i < seq_len; ++i) {
            const float* q_vec = q + (i * num_heads + h) * head_dim;
            float* out_vec = out + (i * num_heads + h) * head_dim;

            std::vector<float> scores(i + 1, 0.0f);
            float max_score = -1e30f;

            // Causal masking: attend only to positions <= i
            for (int32_t j = 0; j <= i; ++j) {
                const float* k_vec = k + (j * num_kv_heads + kv_h) * head_dim;
                float dot = 0.0f;
                for (int32_t d = 0; d < head_dim; ++d) {
                    dot += q_vec[d] * k_vec[d];
                }
                scores[j] = dot * scale;
                if (scores[j] > max_score) {
                    max_score = scores[j];
                }
            }

            // Softmax
            float sum_exp = 0.0f;
            for (int32_t j = 0; j <= i; ++j) {
                scores[j] = std::exp(scores[j] - max_score);
                sum_exp += scores[j];
            }
            float inv_sum = 1.0f / sum_exp;

            // Output projection
            std::memset(out_vec, 0, head_dim * sizeof(float));
            for (int32_t j = 0; j <= i; ++j) {
                const float* v_vec = v + (j * num_kv_heads + kv_h) * head_dim;
                float w = scores[j] * inv_sum;
                for (int32_t d = 0; d < head_dim; ++d) {
                    out_vec[d] += w * v_vec[d];
                }
            }
        }
    }
}

} // namespace tensorlite

#include "tensorlite/model/model_runner.hpp"
#include <cmath>
#include <cstring>

namespace tensorlite {

std::vector<float> ModelRunner::forward_prefill(
    Sequence& seq,
    KVCacheManager& kv_manager,
    int32_t start_token_idx,
    int32_t num_tokens_to_compute
) {
    const auto& cfg = weights_.config;
    const int32_t hidden_size = cfg.hidden_size;
    const int32_t num_heads = cfg.num_attention_heads;
    const int32_t num_kv_heads = cfg.num_key_value_heads;
    const int32_t head_dim = cfg.head_dim;
    const int32_t block_size = kv_manager.config().block_size;
    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const auto& tokens = seq.tokens();
    const auto& block_table = seq.block_table();

    // 1. Embedding lookup for prompt chunk: [num_tokens, hidden_size]
    std::vector<float> hidden_states(num_tokens_to_compute * hidden_size, 0.0f);
    const float* embed_table = static_cast<const float*>(weights_.embed_tokens.data.raw_ptr());

    for (int32_t i = 0; i < num_tokens_to_compute; ++i) {
        TokenId tid = tokens[start_token_idx + i];
        int32_t tid_val = std::clamp(tid.get(), 0, cfg.vocab_size - 1);
        std::memcpy(
            hidden_states.data() + i * hidden_size,
            embed_table + tid_val * hidden_size,
            hidden_size * sizeof(float)
        );
    }

    // Allocate scratch buffers
    std::vector<float> normed_in(num_tokens_to_compute * hidden_size);
    std::vector<float> q(num_tokens_to_compute * num_heads * head_dim);
    std::vector<float> k(num_tokens_to_compute * num_kv_heads * head_dim);
    std::vector<float> v(num_tokens_to_compute * num_kv_heads * head_dim);
    std::vector<float> attn_out(num_tokens_to_compute * num_heads * head_dim);
    std::vector<float> proj_out(num_tokens_to_compute * hidden_size);

    std::vector<float> gate(num_tokens_to_compute * cfg.intermediate_size);
    std::vector<float> up(num_tokens_to_compute * cfg.intermediate_size);
    std::vector<float> swiglu_out(num_tokens_to_compute * cfg.intermediate_size);
    std::vector<float> mlp_out(num_tokens_to_compute * hidden_size);

    // 2. Transformer layers forward
    for (int32_t l = 0; l < cfg.num_hidden_layers; ++l) {
        const auto& layer = weights_.layers[l];

        // 2.1 Input Layernorm
        Ops::rms_norm(
            normed_in.data(),
            hidden_states.data(),
            static_cast<const float*>(layer.input_layernorm.data.raw_ptr()),
            num_tokens_to_compute,
            hidden_size,
            cfg.rms_norm_eps
        );

        // 2.2 Q, K, V Linear projections
        Ops::linear(q.data(), normed_in.data(), layer.q_proj, num_tokens_to_compute, num_heads * head_dim, hidden_size);
        Ops::linear(k.data(), normed_in.data(), layer.k_proj, num_tokens_to_compute, num_kv_heads * head_dim, hidden_size);
        Ops::linear(v.data(), normed_in.data(), layer.v_proj, num_tokens_to_compute, num_kv_heads * head_dim, hidden_size);

        // 2.3 Rotary Position Embeddings (RoPE)
        Ops::apply_rotary_emb(q.data(), num_tokens_to_compute, num_heads, head_dim, start_token_idx, cfg.rope_theta);
        Ops::apply_rotary_emb(k.data(), num_tokens_to_compute, num_kv_heads, head_dim, start_token_idx, cfg.rope_theta);

        // 2.4 Store K and V into KV Cache Manager physical blocks
        for (int32_t i = 0; i < num_tokens_to_compute; ++i) {
            int32_t abs_pos = start_token_idx + i;
            int32_t logical_block = abs_pos / block_size;
            int32_t block_offset = abs_pos % block_size;
            BlockId bid = block_table[logical_block];

            uint8_t* k_blk_ptr = static_cast<uint8_t*>(kv_manager.get_block_ptr(l, bid, false));
            uint8_t* v_blk_ptr = static_cast<uint8_t*>(kv_manager.get_block_ptr(l, bid, true));

            size_t token_offset = static_cast<size_t>(block_offset) * (num_kv_heads * head_dim);

            if (kv_manager.config().dtype == DataType::FP8_E4M3) {
                fp8_e4m3* k_dst = reinterpret_cast<fp8_e4m3*>(k_blk_ptr) + token_offset;
                fp8_e4m3* v_dst = reinterpret_cast<fp8_e4m3*>(v_blk_ptr) + token_offset;
                const float* k_src = k.data() + i * (num_kv_heads * head_dim);
                const float* v_src = v.data() + i * (num_kv_heads * head_dim);

                for (int32_t d = 0; d < num_kv_heads * head_dim; ++d) {
                    k_dst[d] = fp8_e4m3::from_float(k_src[d]);
                    v_dst[d] = fp8_e4m3::from_float(v_src[d]);
                }
            }
        }

        // 2.5 Causal Attention
        AttentionKernel::causal_attention_prefill(
            attn_out.data(),
            q.data(),
            k.data(),
            v.data(),
            num_tokens_to_compute,
            num_heads,
            num_kv_heads,
            head_dim,
            attn_scale
        );

        // 2.6 Out projection and residual connection
        Ops::linear(proj_out.data(), attn_out.data(), layer.o_proj, num_tokens_to_compute, hidden_size, num_heads * head_dim);
        Ops::add_residual(hidden_states.data(), proj_out.data(), num_tokens_to_compute * hidden_size);

        // 2.7 Post-attention Layernorm
        Ops::rms_norm(
            normed_in.data(),
            hidden_states.data(),
            static_cast<const float*>(layer.post_attention_layernorm.data.raw_ptr()),
            num_tokens_to_compute,
            hidden_size,
            cfg.rms_norm_eps
        );

        // 2.8 SwiGLU MLP: Gate, Up, Down Projections
        Ops::linear(gate.data(), normed_in.data(), layer.gate_proj, num_tokens_to_compute, cfg.intermediate_size, hidden_size);
        Ops::linear(up.data(), normed_in.data(), layer.up_proj, num_tokens_to_compute, cfg.intermediate_size, hidden_size);
        Ops::swiglu(swiglu_out.data(), gate.data(), up.data(), num_tokens_to_compute * cfg.intermediate_size);
        Ops::linear(mlp_out.data(), swiglu_out.data(), layer.down_proj, num_tokens_to_compute, hidden_size, cfg.intermediate_size);

        // 2.9 Residual connection
        Ops::add_residual(hidden_states.data(), mlp_out.data(), num_tokens_to_compute * hidden_size);
    }

    // 3. Final RMSNorm
    std::vector<float> final_normed(hidden_size);
    const float* last_hidden = hidden_states.data() + (num_tokens_to_compute - 1) * hidden_size;
    Ops::rms_norm(
        final_normed.data(),
        last_hidden,
        static_cast<const float*>(weights_.final_norm.data.raw_ptr()),
        1,
        hidden_size,
        cfg.rms_norm_eps
    );

    // 4. Compute Logits [vocab_size]
    std::vector<float> logits(cfg.vocab_size, 0.0f);
    if (!weights_.lm_head.data.empty()) {
        Ops::linear(logits.data(), final_normed.data(), weights_.lm_head, 1, cfg.vocab_size, hidden_size);
    } else {
        // Tied embeddings fallback
        Ops::linear(logits.data(), final_normed.data(), weights_.embed_tokens, 1, cfg.vocab_size, hidden_size);
    }

    return logits;
}

std::vector<std::vector<float>> ModelRunner::forward_decode_batch(
    std::vector<std::shared_ptr<Sequence>>& batch,
    KVCacheManager& kv_manager
) {
    const auto& cfg = weights_.config;
    const int32_t num_seqs = static_cast<int32_t>(batch.size());
    if (num_seqs == 0) return {};

    const int32_t hidden_size = cfg.hidden_size;
    const int32_t num_heads = cfg.num_attention_heads;
    const int32_t num_kv_heads = cfg.num_key_value_heads;
    const int32_t head_dim = cfg.head_dim;
    const int32_t block_size = kv_manager.config().block_size;
    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Prepare batch metadata
    std::vector<int32_t> seq_lens(num_seqs);
    int32_t max_blocks = 0;
    for (int32_t s = 0; s < num_seqs; ++s) {
        seq_lens[s] = static_cast<int32_t>(batch[s]->total_len());
        max_blocks = std::max(max_blocks, static_cast<int32_t>(batch[s]->block_table().size()));
    }
    if (max_blocks == 0) max_blocks = 1;

    std::vector<int32_t> block_tables(num_seqs * max_blocks, -1);
    for (int32_t s = 0; s < num_seqs; ++s) {
        const auto& tbl = batch[s]->block_table();
        for (size_t b = 0; b < tbl.size(); ++b) {
            block_tables[s * max_blocks + b] = tbl[b].get();
        }
    }

    // 1. Embedding lookup for the single newest token of each sequence
    std::vector<float> hidden_states(num_seqs * hidden_size);
    const float* embed_table = static_cast<const float*>(weights_.embed_tokens.data.raw_ptr());
    for (int32_t s = 0; s < num_seqs; ++s) {
        TokenId tid = batch[s]->last_token();
        int32_t tid_val = std::clamp(tid.get(), 0, cfg.vocab_size - 1);
        std::memcpy(
            hidden_states.data() + s * hidden_size,
            embed_table + tid_val * hidden_size,
            hidden_size * sizeof(float)
        );
    }

    // Scratch buffers
    std::vector<float> normed_in(num_seqs * hidden_size);
    std::vector<float> q(num_seqs * num_heads * head_dim);
    std::vector<float> k(num_seqs * num_kv_heads * head_dim);
    std::vector<float> v(num_seqs * num_kv_heads * head_dim);
    std::vector<float> attn_out(num_seqs * num_heads * head_dim);
    std::vector<float> proj_out(num_seqs * hidden_size);

    std::vector<float> gate(num_seqs * cfg.intermediate_size);
    std::vector<float> up(num_seqs * cfg.intermediate_size);
    std::vector<float> swiglu_out(num_seqs * cfg.intermediate_size);
    std::vector<float> mlp_out(num_seqs * hidden_size);

    PagedAttentionParams attn_params;
    attn_params.num_seqs = num_seqs;
    attn_params.num_heads = num_heads;
    attn_params.num_kv_heads = num_kv_heads;
    attn_params.head_dim = head_dim;
    attn_params.block_size = block_size;
    attn_params.scale = attn_scale;
    attn_params.kv_cache_scale = 1.0f;

    // 2. Transformer layers forward
    for (int32_t l = 0; l < cfg.num_hidden_layers; ++l) {
        const auto& layer = weights_.layers[l];

        // 2.1 Norm
        Ops::rms_norm(normed_in.data(), hidden_states.data(), static_cast<const float*>(layer.input_layernorm.data.raw_ptr()), num_seqs, hidden_size, cfg.rms_norm_eps);

        // 2.2 Projections
        Ops::linear(q.data(), normed_in.data(), layer.q_proj, num_seqs, num_heads * head_dim, hidden_size);
        Ops::linear(k.data(), normed_in.data(), layer.k_proj, num_seqs, num_kv_heads * head_dim, hidden_size);
        Ops::linear(v.data(), normed_in.data(), layer.v_proj, num_seqs, num_kv_heads * head_dim, hidden_size);

        // 2.3 RoPE
        for (int32_t s = 0; s < num_seqs; ++s) {
            int32_t pos = seq_lens[s] - 1;
            Ops::apply_rotary_emb(q.data() + s * (num_heads * head_dim), 1, num_heads, head_dim, pos, cfg.rope_theta);
            Ops::apply_rotary_emb(k.data() + s * (num_kv_heads * head_dim), 1, num_kv_heads, head_dim, pos, cfg.rope_theta);
        }

        // 2.4 Store newly generated token's K and V into KV cache
        for (int32_t s = 0; s < num_seqs; ++s) {
            int32_t pos = seq_lens[s] - 1;
            int32_t logical_block = pos / block_size;
            int32_t block_offset = pos % block_size;
            BlockId bid = batch[s]->block_table()[logical_block];

            uint8_t* k_blk_ptr = static_cast<uint8_t*>(kv_manager.get_block_ptr(l, bid, false));
            uint8_t* v_blk_ptr = static_cast<uint8_t*>(kv_manager.get_block_ptr(l, bid, true));
            size_t token_offset = static_cast<size_t>(block_offset) * (num_kv_heads * head_dim);

            fp8_e4m3* k_dst = reinterpret_cast<fp8_e4m3*>(k_blk_ptr) + token_offset;
            fp8_e4m3* v_dst = reinterpret_cast<fp8_e4m3*>(v_blk_ptr) + token_offset;
            const float* k_src = k.data() + s * (num_kv_heads * head_dim);
            const float* v_src = v.data() + s * (num_kv_heads * head_dim);

            for (int32_t d = 0; d < num_kv_heads * head_dim; ++d) {
                k_dst[d] = fp8_e4m3::from_float(k_src[d]);
                v_dst[d] = fp8_e4m3::from_float(v_src[d]);
            }
        }

        // 2.5 PagedAttention Decode Kernel across entire continuous batch!
        AttentionKernel::paged_attention_decode(
            attn_out.data(),
            q.data(),
            kv_manager.get_block_ptr(l, BlockId{0}, false), // Key pool base for this layer
            kv_manager.get_block_ptr(l, BlockId{0}, true),  // Value pool base for this layer
            block_tables.data(),
            seq_lens.data(),
            max_blocks,
            attn_params,
            true
        );

        // 2.6 Out projection and residual
        Ops::linear(proj_out.data(), attn_out.data(), layer.o_proj, num_seqs, hidden_size, num_heads * head_dim);
        Ops::add_residual(hidden_states.data(), proj_out.data(), num_seqs * hidden_size);

        // 2.7 Post norm & MLP
        Ops::rms_norm(normed_in.data(), hidden_states.data(), static_cast<const float*>(layer.post_attention_layernorm.data.raw_ptr()), num_seqs, hidden_size, cfg.rms_norm_eps);
        Ops::linear(gate.data(), normed_in.data(), layer.gate_proj, num_seqs, cfg.intermediate_size, hidden_size);
        Ops::linear(up.data(), normed_in.data(), layer.up_proj, num_seqs, cfg.intermediate_size, hidden_size);
        Ops::swiglu(swiglu_out.data(), gate.data(), up.data(), num_seqs * cfg.intermediate_size);
        Ops::linear(mlp_out.data(), swiglu_out.data(), layer.down_proj, num_seqs, hidden_size, cfg.intermediate_size);
        Ops::add_residual(hidden_states.data(), mlp_out.data(), num_seqs * hidden_size);
    }

    // 3. Final RMSNorm and Logits
    std::vector<std::vector<float>> all_logits(num_seqs, std::vector<float>(cfg.vocab_size));
    std::vector<float> final_normed(num_seqs * hidden_size);
    Ops::rms_norm(final_normed.data(), hidden_states.data(), static_cast<const float*>(weights_.final_norm.data.raw_ptr()), num_seqs, hidden_size, cfg.rms_norm_eps);

    for (int32_t s = 0; s < num_seqs; ++s) {
        const float* norm_row = final_normed.data() + s * hidden_size;
        if (!weights_.lm_head.data.empty()) {
            Ops::linear(all_logits[s].data(), norm_row, weights_.lm_head, 1, cfg.vocab_size, hidden_size);
        } else {
            Ops::linear(all_logits[s].data(), norm_row, weights_.embed_tokens, 1, cfg.vocab_size, hidden_size);
        }
    }

    return all_logits;
}

} // namespace tensorlite

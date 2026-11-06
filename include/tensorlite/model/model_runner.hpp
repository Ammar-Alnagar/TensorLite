#pragma once

#include <vector>
#include <memory>
#include "tensorlite/model/model_config.hpp"
#include "tensorlite/model/model_weights.hpp"
#include "tensorlite/model/ops.hpp"
#include "tensorlite/cache/kv_cache_manager.hpp"
#include "tensorlite/kernel/attention.hpp"

namespace tensorlite {

// Forward pass execution engine for full Qwen / LLaMA transformer architecture
class ModelRunner {
public:
    explicit ModelRunner(ModelWeights weights) : weights_(std::move(weights)) {}

    [[nodiscard]] const ModelConfig& config() const noexcept { return weights_.config; }
    [[nodiscard]] const ModelWeights& weights() const noexcept { return weights_; }

    // Prefill step for a single prompt sequence: computes KV projections, stores into
    // physical blocks managed by kv_manager, and returns next-token logits [vocab_size]
    std::vector<float> forward_prefill(
        Sequence& seq,
        KVCacheManager& kv_manager,
        int32_t start_token_idx,
        int32_t num_tokens_to_compute
    );

    // Decode step for a continuous batch of sequences:
    // Performs PagedAttention for all sequences simultaneously and returns logits for each sequence
    std::vector<std::vector<float>> forward_decode_batch(
        std::vector<std::shared_ptr<Sequence>>& batch,
        KVCacheManager& kv_manager
    );

private:
    ModelWeights weights_;
};

} // namespace tensorlite

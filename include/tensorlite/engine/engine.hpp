#pragma once

#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include "tensorlite/engine/tokenizer.hpp"
#include "tensorlite/engine/request.hpp"
#include "tensorlite/engine/sampler.hpp"
#include "tensorlite/model/model_runner.hpp"
#include "tensorlite/cache/kv_cache_manager.hpp"
#include "tensorlite/cache/prefix_cache.hpp"
#include "tensorlite/scheduler/scheduler.hpp"

namespace tensorlite {

// Top-level LLM Inference Engine orchestrating the entire runtime lifecycle
class InferenceEngine {
public:
    InferenceEngine(
        ModelWeights weights,
        CacheConfig cache_config,
        SchedulerConfig sched_config
    );

    // Add a new user prompt request to the engine
    void add_request(
        const std::string& request_id,
        const std::string& prompt,
        const SamplingParams& params,
        TokenStreamCallback callback = nullptr
    );

    // Advance engine by exactly one iteration:
    // Scheduler -> Prefill / PagedAttention Decode -> Sample -> Stream Tokens
    // Returns true if there is ongoing work, false if idle
    bool step();

    [[nodiscard]] Tokenizer& tokenizer() noexcept { return tokenizer_; }
    [[nodiscard]] const Tokenizer& tokenizer() const noexcept { return tokenizer_; }

    [[nodiscard]] size_t num_waiting() const noexcept { return scheduler_->num_waiting(); }
    [[nodiscard]] size_t num_running() const noexcept { return scheduler_->num_running(); }
    [[nodiscard]] bool has_work() const noexcept { return scheduler_->has_work(); }

    [[nodiscard]] std::shared_ptr<KVCacheManager> kv_cache_manager() noexcept { return kv_manager_; }
    [[nodiscard]] std::shared_ptr<PrefixCache> prefix_cache() noexcept { return prefix_cache_; }

private:
    void handle_sampled_token(Sequence& seq, TokenId next_token, const Request& req);

    Tokenizer tokenizer_;
    std::shared_ptr<KVCacheManager> kv_manager_;
    std::shared_ptr<PrefixCache> prefix_cache_;
    std::shared_ptr<Scheduler> scheduler_;
    std::unique_ptr<ModelRunner> runner_;
    Sampler sampler_;
    std::atomic<int64_t> sequence_counter_{0};
};

} // namespace tensorlite

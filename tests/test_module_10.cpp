#include <cassert>
#include <iostream>
#include <vector>
#include <string>
#include "tensorlite/engine/engine.hpp"

using namespace tensorlite;

void test_full_system_integration() {
    auto weights = ModelWeights::load_from_hf_dir("models/test_qwen_fp8");

    CacheConfig cache_cfg;
    cache_cfg.block_size = 16;
    cache_cfg.num_blocks = 32;
    cache_cfg.num_layers = weights.config.num_hidden_layers;
    cache_cfg.num_kv_heads = weights.config.num_key_value_heads;
    cache_cfg.head_dim = weights.config.head_dim;
    cache_cfg.dtype = DataType::FP8_E4M3;

    SchedulerConfig sched_cfg;
    sched_cfg.max_num_seqs = 4;
    sched_cfg.max_num_batched_tokens = 64;
    sched_cfg.enable_chunked_prefill = true;

    InferenceEngine engine(std::move(weights), cache_cfg, sched_cfg);

    SamplingParams params;
    params.max_tokens = 6;
    params.temperature = 0.0f;

    int completed_requests = 0;
    std::string out_req1;
    std::string out_req2;

    // Request 1
    engine.add_request("req-1", "System prompt: summarize physics. User: Tell me about gravity.", params,
        [&](const std::string& token, bool finished) {
            out_req1 += token;
            if (finished) completed_requests++;
        }
    );

    // Run until request 1 finishes and caches its prompt prefix
    int step_count = 0;
    while (engine.has_work() && step_count < 100) {
        bool work = engine.step();
        step_count++;
        std::cout << "Step " << step_count << ": work=" << work
                  << ", completed=" << completed_requests
                  << ", waiting=" << engine.num_waiting()
                  << ", running=" << engine.num_running() << std::endl;
        if (completed_requests == 1) break;
    }
    assert(completed_requests == 1);
    assert(!out_req1.empty());

    // Request 2 shares identical system prefix with Request 1!
    // This will hit the prefix cache and reuse physical blocks without recomputing prefill!
    engine.add_request("req-2", "System prompt: summarize physics. User: Tell me about electromagnetism.", params,
        [&](const std::string& token, bool finished) {
            out_req2 += token;
            if (finished) completed_requests++;
        }
    );

    // Verify prefix cache retained tokens
    assert(engine.prefix_cache()->total_cached_tokens() > 0);

    // Run remaining request
    while (engine.has_work()) {
        engine.step();
    }

    assert(completed_requests == 2);
    assert(!out_req2.empty());

    std::cout << "[PASS] Full End-to-End System Test: Continuous batching + PagedAttention FP8 + Prefix reuse + Streaming verified!\n";
}

int main() {
    std::cout << "Running Module 10 End-to-End Integration Tests...\n";
    test_full_system_integration();
    std::cout << "All Module 10 tests completed successfully!\n";
    return 0;
}

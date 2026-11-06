#include <cassert>
#include <iostream>
#include <vector>
#include <string>
#include "tensorlite/engine/engine.hpp"

using namespace tensorlite;

void test_engine_end_to_end_streaming() {
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
    params.max_tokens = 5;
    params.temperature = 0.0f; // Deterministic greedy

    std::string stream_out_1;
    bool finished_1 = false;
    engine.add_request("req-1", "Prompt alpha", params, [&](const std::string& token, bool done) {
        stream_out_1 += token;
        if (done) finished_1 = true;
    });

    std::string stream_out_2;
    bool finished_2 = false;
    engine.add_request("req-2", "Prompt beta", params, [&](const std::string& token, bool done) {
        stream_out_2 += token;
        if (done) finished_2 = true;
    });

    assert(engine.num_waiting() == 2);
    assert(engine.has_work());

    // Run engine loop until all requests finish
    int steps = 0;
    while (engine.step() && steps < 50) {
        steps++;
    }

    assert(finished_1);
    assert(finished_2);
    assert(!engine.has_work());

    std::cout << "[PASS] Engine loop: continuous batching + token streaming verified in " << steps << " steps!\n";
}

int main() {
    std::cout << "Running Module 9 Tests...\n";
    test_engine_end_to_end_streaming();
    std::cout << "All Module 9 tests completed successfully!\n";
    return 0;
}

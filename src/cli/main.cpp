#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <filesystem>
#include "tensorlite/engine/engine.hpp"

using namespace tensorlite;

void print_banner() {
    std::cout << "=================================================================\n";
    std::cout << "  TensorLite: High-Performance Educational C++20/CUDA LLM Engine  \n";
    std::cout << "  Features: PagedAttention | FP8 KV Cache | Continuous Batching    \n";
    std::cout << "            Prefix Caching (Radix) | Streaming Generation         \n";
    std::cout << "=================================================================\n\n";
}

int main(int argc, char** argv) {
    print_banner();

    std::filesystem::path model_dir = "models/test_qwen_fp8";
    if (argc > 1) {
        model_dir = argv[1];
    }

    if (!std::filesystem::exists(model_dir / "config.json")) {
        std::cerr << "Error: Model path " << model_dir << " does not contain config.json.\n";
        std::cerr << "Run: python3 scripts/create_test_model.py " << model_dir << "\n";
        return 1;
    }

    std::cout << "[1/4] Loading FP8 weights from: " << model_dir.string() << "...\n";
    auto weights = ModelWeights::load_from_hf_dir(model_dir);
    std::cout << "  - Model type: " << weights.config.model_type << "\n";
    std::cout << "  - Layers: " << weights.config.num_hidden_layers << "\n";
    std::cout << "  - Attention Heads: " << weights.config.num_attention_heads
              << " (Q) / " << weights.config.num_key_value_heads << " (KV)\n";
    std::cout << "  - Hidden Size: " << weights.config.hidden_size << "\n";
    std::cout << "  - Vocab Size: " << weights.config.vocab_size << "\n";

    std::cout << "\n[2/4] Initializing Paged KV Cache and Scheduler...\n";
    CacheConfig cache_cfg;
    cache_cfg.block_size = 16;
    cache_cfg.num_blocks = 64;
    cache_cfg.num_layers = weights.config.num_hidden_layers;
    cache_cfg.num_kv_heads = weights.config.num_key_value_heads;
    cache_cfg.head_dim = weights.config.head_dim;
    cache_cfg.dtype = DataType::FP8_E4M3;

    SchedulerConfig sched_cfg;
    sched_cfg.max_num_seqs = 8;
    sched_cfg.max_num_batched_tokens = 128;
    sched_cfg.enable_chunked_prefill = true;

    InferenceEngine engine(std::move(weights), cache_cfg, sched_cfg);

    std::cout << "\n[3/4] Enqueueing Concurrent Requests for Continuous Batching Demo...\n";

    SamplingParams params;
    params.max_tokens = 8;
    params.temperature = 0.7f;
    params.top_p = 0.9f;

    std::vector<std::string> prompts = {
        "User 1: Explain quantum physics in one sentence.",
        "User 2: Write a C++ function to reverse a string.",
        "User 1: Explain quantum computing in one sentence." // Shares prefix with User 1!
    };

    for (size_t i = 0; i < prompts.size(); ++i) {
        std::string req_id = std::format("req-{}", i + 1);
        std::cout << "  + Added [" << req_id << "]: \"" << prompts[i] << "\"\n";

        engine.add_request(
            req_id,
            prompts[i],
            params,
            [req_id](const std::string& token, bool finished) {
                if (!token.empty()) {
                    std::cout << "[" << req_id << " token]: " << token << "\n";
                }
                if (finished) {
                    std::cout << "[" << req_id << "]: Generation complete.\n";
                }
            }
        );
    }

    std::cout << "\n[4/4] Executing Engine Iteration Loop (Continuous Batching + Streaming)...\n";
    auto start_time = std::chrono::steady_clock::now();

    int iteration = 0;
    while (engine.has_work()) {
        iteration++;
        engine.step();
    }

    auto end_time = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    std::cout << "\n=================================================================\n";
    std::cout << "All requests finished successfully!\n";
    std::cout << "  - Total Iterations: " << iteration << "\n";
    std::cout << "  - Total Elapsed Time: " << elapsed_ms << " ms\n";
    std::cout << "  - Prefix Cache Total Tokens Reused: "
              << engine.prefix_cache()->total_cached_tokens() << " tokens\n";
    std::cout << "=================================================================\n";

    return 0;
}

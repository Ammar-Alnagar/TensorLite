#include <cassert>
#include <iostream>
#include <vector>
#include <memory>
#include "tensorlite/model/model_runner.hpp"
#include "tensorlite/cache/kv_cache_manager.hpp"

using namespace tensorlite;

void test_transformer_prefill_and_decode() {
    // 1. Load the synthetic FP8 Qwen model
    auto weights = ModelWeights::load_from_hf_dir("models/test_qwen_fp8");
    ModelRunner runner(std::move(weights));

    // 2. Set up KV Cache Manager
    CacheConfig cache_cfg;
    cache_cfg.block_size = 16;
    cache_cfg.num_blocks = 32;
    cache_cfg.num_layers = runner.config().num_hidden_layers;
    cache_cfg.num_kv_heads = runner.config().num_key_value_heads;
    cache_cfg.head_dim = runner.config().head_dim;
    cache_cfg.dtype = DataType::FP8_E4M3;

    KVCacheManager kv_mgr(cache_cfg);

    // 3. Test Prefill on a sequence with 10 prompt tokens
    std::vector<TokenId> prompt_tokens;
    for (int i = 0; i < 10; ++i) {
        prompt_tokens.push_back(TokenId{i + 5});
    }

    auto seq = std::make_shared<Sequence>(SequenceId{1}, prompt_tokens);
    kv_mgr.allocate_for_sequence(*seq);
    assert(seq->block_table().size() == 1);

    auto logits_prefill = runner.forward_prefill(*seq, kv_mgr, 0, 10);
    assert(logits_prefill.size() == static_cast<size_t>(runner.config().vocab_size));
    seq->set_num_computed_tokens(10);
    seq->set_status(SequenceStatus::RUNNING);

    std::cout << "[PASS] Prefill forward pass completed, logits dimension: " << logits_prefill.size() << "\n";

    // 4. Test Decode on a batch of 2 sequences
    // Add a generated token to seq 1
    seq->append_token(TokenId{42});

    // Create a second sequence with 5 prompt tokens prefilled
    std::vector<TokenId> prompt2 = {TokenId{1}, TokenId{2}, TokenId{3}, TokenId{4}, TokenId{5}};
    auto seq2 = std::make_shared<Sequence>(SequenceId{2}, prompt2);
    kv_mgr.allocate_for_sequence(*seq2);
    runner.forward_prefill(*seq2, kv_mgr, 0, 5);
    seq2->set_num_computed_tokens(5);
    seq2->set_status(SequenceStatus::RUNNING);
    seq2->append_token(TokenId{99});

    std::vector<std::shared_ptr<Sequence>> decode_batch = {seq, seq2};
    auto batch_logits = runner.forward_decode_batch(decode_batch, kv_mgr);

    assert(batch_logits.size() == 2);
    assert(batch_logits[0].size() == static_cast<size_t>(runner.config().vocab_size));
    assert(batch_logits[1].size() == static_cast<size_t>(runner.config().vocab_size));

    std::cout << "[PASS] Batch Decode with PagedAttention forward pass completed.\n";
}

int main() {
    std::cout << "Running Module 6 Tests...\n";
    test_transformer_prefill_and_decode();
    std::cout << "All Module 6 tests completed successfully!\n";
    return 0;
}

#include <cassert>
#include <iostream>
#include <vector>
#include <memory>
#include "tensorlite/cache/prefix_cache.hpp"

using namespace tensorlite;

void test_radix_prefix_cache_reuse() {
    CacheConfig cache_cfg;
    cache_cfg.block_size = 16;
    cache_cfg.num_blocks = 20;
    auto kv_mgr = std::make_shared<KVCacheManager>(cache_cfg);

    PrefixCache prefix_cache(kv_mgr);

    // Prompt A has 32 tokens (exactly 2 physical blocks)
    std::vector<TokenId> prompt_a;
    for (int i = 0; i < 32; ++i) {
        prompt_a.push_back(TokenId{i + 10});
    }

    // Allocate 2 blocks for Prompt A
    BlockId b0 = kv_mgr->allocate_block();
    BlockId b1 = kv_mgr->allocate_block();
    std::vector<BlockId> blocks_a = {b0, b1};

    // Insert Prompt A into Prefix Cache
    prefix_cache.insert(prompt_a, blocks_a);

    // Verify blocks retain count inside prefix cache
    assert(kv_mgr->get_ref_count(b0) == 2); // 1 for prompt_a owner + 1 for radix tree node
    assert(kv_mgr->get_ref_count(b1) == 2);

    // Prompt B shares the first 32 tokens with Prompt A, followed by 10 new tokens
    std::vector<TokenId> prompt_b = prompt_a;
    for (int i = 0; i < 10; ++i) {
        prompt_b.push_back(TokenId{999 + i});
    }

    // Match prefix for Prompt B
    auto match = prefix_cache.match_prefix(prompt_b);
    assert(match.matched_tokens == 32);
    assert(match.reusable_blocks.size() == 2);
    assert(match.reusable_blocks[0] == b0);
    assert(match.reusable_blocks[1] == b1);

    // Retained for Prompt B: ref_count now 3
    assert(kv_mgr->get_ref_count(b0) == 3);
    assert(kv_mgr->get_ref_count(b1) == 3);

    // Clean up
    kv_mgr->release_block(b0); // Release prompt_a owner
    kv_mgr->release_block(b1);
    kv_mgr->release_block(match.reusable_blocks[0]); // Release prompt_b owner
    kv_mgr->release_block(match.reusable_blocks[1]);

    // Only radix tree retains them now
    assert(kv_mgr->get_ref_count(b0) == 1);
    assert(kv_mgr->get_ref_count(b1) == 1);

    std::cout << "[PASS] Radix Prefix Cache longest-prefix matching & zero-overhead block reuse verified.\n";
}

int main() {
    std::cout << "Running Module 8 Tests...\n";
    test_radix_prefix_cache_reuse();
    std::cout << "All Module 8 tests completed successfully!\n";
    return 0;
}

#include <cassert>
#include <iostream>
#include <vector>
#include "tensorlite/cache/kv_cache_manager.hpp"

using namespace tensorlite;

void test_kv_cache_allocation_free() {
    CacheConfig config;
    config.block_size = 16;
    config.num_blocks = 10;
    config.num_layers = 2;
    config.num_kv_heads = 2;
    config.head_dim = 16;
    config.dtype = DataType::FP8_E4M3;

    KVCacheManager manager(config);
    assert(manager.num_total_blocks() == 10);
    assert(manager.num_free_blocks() == 10);

    // Allocate 3 blocks
    BlockId b0 = manager.allocate_block();
    BlockId b1 = manager.allocate_block();
    BlockId b2 = manager.allocate_block();

    assert(manager.num_free_blocks() == 7);
    assert(manager.get_ref_count(b0) == 1);
    assert(manager.get_ref_count(b1) == 1);

    // Test retain (shared block)
    manager.retain_block(b0);
    assert(manager.get_ref_count(b0) == 2);

    // Release once
    manager.release_block(b0);
    assert(manager.get_ref_count(b0) == 1);
    assert(manager.num_free_blocks() == 7);

    // Release second time -> returns to free pool
    manager.release_block(b0);
    assert(manager.num_free_blocks() == 8);

    manager.release_block(b1);
    manager.release_block(b2);
    assert(manager.num_free_blocks() == 10);

    std::cout << "[PASS] KVCacheManager block allocation, retain, and free verified.\n";
}

void test_sequence_block_table_mapping() {
    CacheConfig config;
    config.block_size = 16;
    config.num_blocks = 8;
    config.num_layers = 2;
    config.num_kv_heads = 2;
    config.head_dim = 16;

    KVCacheManager manager(config);

    // Create a sequence with 35 tokens -> needs ceil(35 / 16) = 3 blocks
    std::vector<TokenId> prompt(35, TokenId{1});
    Sequence seq(SequenceId{1}, prompt);

    manager.allocate_for_sequence(seq);
    assert(seq.block_table().size() == 3);
    assert(manager.num_free_blocks() == 5);

    // Append tokens to exceed 48 tokens -> needs 4th block
    for (int i = 0; i < 14; ++i) {
        seq.append_token(TokenId{2});
    }
    assert(seq.total_len() == 49);
    manager.allocate_for_sequence(seq);
    assert(seq.block_table().size() == 4);
    assert(manager.num_free_blocks() == 4);

    // Free sequence
    manager.free_sequence(seq);
    assert(seq.block_table().empty());
    assert(manager.num_free_blocks() == 8);

    std::cout << "[PASS] Dynamic sequence block table growth and release verified.\n";
}

int main() {
    std::cout << "Running Module 4 Tests...\n";
    test_kv_cache_allocation_free();
    test_sequence_block_table_mapping();
    std::cout << "All Module 4 tests completed successfully!\n";
    return 0;
}

#pragma once

#include <vector>
#include <deque>
#include <unordered_map>
#include <memory>
#include <cassert>
#include "tensorlite/core/types.hpp"
#include "tensorlite/core/device_buffer.hpp"
#include "tensorlite/core/error.hpp"
#include "tensorlite/engine/request.hpp"

namespace tensorlite {

// Configuration for Paged KV Cache
struct CacheConfig {
    int32_t block_size{16};          // Number of tokens per physical block (typically 16 or 32)
    int32_t num_blocks{128};         // Total physical blocks pre-allocated in pool
    int32_t num_layers{2};           // Number of transformer layers
    int32_t num_kv_heads{2};         // Key-Value heads (GQA)
    int32_t head_dim{16};            // Dimension per head
    DataType dtype{DataType::FP8_E4M3}; // FP8 E4M3 for KV cache

    [[nodiscard]] size_t bytes_per_token_per_layer() const noexcept {
        // Both Key and Value vectors: 2 * num_kv_heads * head_dim * sizeof(dtype)
        return 2 * num_kv_heads * head_dim * data_type_size(dtype);
    }

    [[nodiscard]] size_t bytes_per_block_per_layer() const noexcept {
        return block_size * bytes_per_token_per_layer();
    }

    [[nodiscard]] size_t total_cache_bytes() const noexcept {
        return num_layers * num_blocks * bytes_per_block_per_layer();
    }
};

// Physical Block metadata tracked on host
struct PhysicalBlock {
    BlockId id{INVALID_BLOCK_ID};
    int32_t ref_count{0};            // Reference counting for prefix sharing / Copy-on-Write
    int32_t num_tokens{0};           // Current filled tokens in this block [0..block_size]
};

// Memory manager for PagedAttention physical blocks and sequence block tables
class KVCacheManager {
public:
    explicit KVCacheManager(CacheConfig config)
        : config_(config),
          physical_blocks_(config.num_blocks),
          // Pre-allocate the giant unified contiguous physical KV cache buffer
          kv_memory_pool_(config.total_cache_bytes()) {

        // Initialize free block list
        for (int32_t i = 0; i < config_.num_blocks; ++i) {
            physical_blocks_[i].id = BlockId{i};
            physical_blocks_[i].ref_count = 0;
            physical_blocks_[i].num_tokens = 0;
            free_blocks_.push_back(BlockId{i});
        }
    }

    [[nodiscard]] const CacheConfig& config() const noexcept { return config_; }
    [[nodiscard]] size_t num_free_blocks() const noexcept { return free_blocks_.size(); }
    [[nodiscard]] size_t num_total_blocks() const noexcept { return config_.num_blocks; }

    // Check if the pool can accommodate allocating `num_needed` blocks
    [[nodiscard]] bool can_allocate(size_t num_needed) const noexcept {
        return free_blocks_.size() >= num_needed;
    }

    // Allocate an empty physical block from the free list
    BlockId allocate_block() {
        if (free_blocks_.empty()) {
            throw OutOfMemoryException("KV Cache exhausted: no free physical blocks available");
        }
        BlockId bid = free_blocks_.front();
        free_blocks_.pop_front();

        auto& blk = physical_blocks_[bid.get()];
        blk.ref_count = 1;
        blk.num_tokens = 0;
        return bid;
    }

    // Increment reference count when a block is shared (e.g. prefix reuse or beam search)
    void retain_block(BlockId bid) {
        if (bid.get() < 0 || bid.get() >= config_.num_blocks) {
            throw InvalidArgumentException("Invalid block id for retain");
        }
        physical_blocks_[bid.get()].ref_count++;
    }

    // Release block: decrement reference count; return to free list if ref_count hits 0
    void release_block(BlockId bid) {
        if (bid.get() < 0 || bid.get() >= config_.num_blocks) {
            throw InvalidArgumentException("Invalid block id for release");
        }
        auto& blk = physical_blocks_[bid.get()];
        assert(blk.ref_count > 0 && "Double-free or corrupted ref_count on block");

        blk.ref_count--;
        if (blk.ref_count == 0) {
            blk.num_tokens = 0;
            free_blocks_.push_back(bid);
        }
    }

    // Allocate blocks for a sequence to accommodate its current token length
    void allocate_for_sequence(Sequence& seq) {
        size_t total_tokens = seq.total_len();
        size_t blocks_needed = (total_tokens + config_.block_size - 1) / config_.block_size;
        auto& table = seq.block_table();

        while (table.size() < blocks_needed) {
            table.push_back(allocate_block());
        }
    }

    // Free all blocks bound to a sequence
    void free_sequence(Sequence& seq) {
        auto& table = seq.block_table();
        for (BlockId bid : table) {
            release_block(bid);
        }
        table.clear();
    }

    // Get pointer to Key or Value slot inside a specific block for a layer
    // Layout within pool: [Layer, Block, Head, Pos, Dim] or [Layer, Block, 2 (K/V), Head, Pos, Dim]
    [[nodiscard]] void* get_block_ptr(int32_t layer_idx, BlockId bid, bool is_value) {
        assert(layer_idx >= 0 && layer_idx < config_.num_layers);
        assert(bid.get() >= 0 && bid.get() < config_.num_blocks);

        size_t layer_offset = layer_idx * (config_.num_blocks * config_.bytes_per_block_per_layer());
        size_t block_offset = bid.get() * config_.bytes_per_block_per_layer();
        // Key is placed in first half of block, Value in second half
        size_t half_block_bytes = config_.bytes_per_block_per_layer() / 2;
        size_t kv_offset = is_value ? half_block_bytes : 0;

        uint8_t* base = static_cast<uint8_t*>(kv_memory_pool_.raw_ptr());
        return base + layer_offset + block_offset + kv_offset;
    }

    [[nodiscard]] DeviceBuffer& raw_pool() noexcept { return kv_memory_pool_; }
    [[nodiscard]] const DeviceBuffer& raw_pool() const noexcept { return kv_memory_pool_; }

    [[nodiscard]] int32_t get_ref_count(BlockId bid) const {
        return physical_blocks_[bid.get()].ref_count;
    }

private:
    CacheConfig config_;
    std::vector<PhysicalBlock> physical_blocks_;
    std::deque<BlockId> free_blocks_;
    DeviceBuffer kv_memory_pool_;
};

} // namespace tensorlite

#pragma once

#include <vector>
#include <unordered_map>
#include <memory>
#include <span>
#include "tensorlite/core/types.hpp"
#include "tensorlite/cache/kv_cache_manager.hpp"

namespace tensorlite {

// Node in the Radix Tree / Prefix Cache
struct PrefixNode {
    std::vector<TokenId> token_prefix;     // Token slice stored at this edge
    std::vector<BlockId> block_ids;        // Physical KV blocks corresponding to this prefix
    std::unordered_map<int32_t, std::shared_ptr<PrefixNode>> children; // Branch on first token of next edge
    std::weak_ptr<PrefixNode> parent;

    PrefixNode() = default;
};

// Radix Prefix Cache managing cross-request KV block sharing
// Mimics SGLang's RadixAttention
class PrefixCache {
public:
    explicit PrefixCache(std::shared_ptr<KVCacheManager> kv_manager)
        : kv_manager_(std::move(kv_manager)),
          root_(std::make_shared<PrefixNode>()) {}

    // Match longest cached prefix for a prompt
    // Returns list of cached BlockIds that can be directly reused without recomputing prefill!
    struct MatchResult {
        size_t matched_tokens{0};
        std::vector<BlockId> reusable_blocks;
    };

    MatchResult match_prefix(std::span<const TokenId> tokens);

    // Insert newly computed token sequence and its corresponding physical blocks into the Radix Tree
    void insert(std::span<const TokenId> tokens, const std::vector<BlockId>& blocks);

    [[nodiscard]] size_t total_cached_tokens() const noexcept { return total_cached_tokens_; }

private:
    std::shared_ptr<KVCacheManager> kv_manager_;
    std::shared_ptr<PrefixNode> root_;
    size_t total_cached_tokens_{0};
};

} // namespace tensorlite

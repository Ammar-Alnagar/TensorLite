#include "tensorlite/cache/prefix_cache.hpp"
#include <algorithm>

namespace tensorlite {

PrefixCache::MatchResult PrefixCache::match_prefix(std::span<const TokenId> tokens) {
    MatchResult result;
    auto current = root_;
    size_t token_offset = 0;
    int32_t block_size = kv_manager_->config().block_size;

    while (token_offset < tokens.size()) {
        int32_t next_token_val = tokens[token_offset].get();
        auto it = current->children.find(next_token_val);
        if (it == current->children.end()) {
            break;
        }

        auto child = it->second;
        const auto& edge_tokens = child->token_prefix;
        size_t match_len = 0;

        while (match_len < edge_tokens.size() &&
               (token_offset + match_len) < tokens.size() &&
               tokens[token_offset + match_len] == edge_tokens[match_len]) {
            match_len++;
        }

        if (match_len == edge_tokens.size()) {
            // Full edge matched
            token_offset += match_len;
            current = child;

            // Retain each reusable physical block
            for (BlockId bid : child->block_ids) {
                kv_manager_->retain_block(bid);
                result.reusable_blocks.push_back(bid);
            }
        } else {
            // Partial match along edge: partial blocks cannot be safely shared without splitting
            break;
        }
    }

    result.matched_tokens = result.reusable_blocks.size() * block_size;
    return result;
}

void PrefixCache::insert(std::span<const TokenId> tokens, const std::vector<BlockId>& blocks) {
    if (tokens.empty() || blocks.empty()) return;

    int32_t block_size = kv_manager_->config().block_size;
    size_t num_full_blocks = blocks.size();

    // Only cache complete physical blocks to guarantee block boundary alignment
    size_t cacheable_tokens = num_full_blocks * block_size;
    if (cacheable_tokens > tokens.size()) {
        cacheable_tokens = (tokens.size() / block_size) * block_size;
        num_full_blocks = cacheable_tokens / block_size;
    }

    if (num_full_blocks == 0) return;

    auto current = root_;
    size_t token_offset = 0;
    size_t block_offset = 0;

    while (token_offset < cacheable_tokens) {
        int32_t next_token_val = tokens[token_offset].get();
        auto it = current->children.find(next_token_val);

        if (it == current->children.end()) {
            // Create a new child node spanning the remainder of the cacheable tokens
            auto new_node = std::make_shared<PrefixNode>();
            new_node->parent = current;

            size_t edge_len = cacheable_tokens - token_offset;
            new_node->token_prefix.assign(tokens.begin() + token_offset, tokens.begin() + cacheable_tokens);

            size_t blocks_for_node = edge_len / block_size;
            for (size_t b = 0; b < blocks_for_node; ++b) {
                BlockId bid = blocks[block_offset + b];
                kv_manager_->retain_block(bid); // Retained inside the radix tree
                new_node->block_ids.push_back(bid);
            }

            current->children[next_token_val] = new_node;
            total_cached_tokens_ += edge_len;
            break;
        }

        // Existing child: advance along common edge
        auto child = it->second;
        size_t edge_len = child->token_prefix.size();
        token_offset += edge_len;
        block_offset += child->block_ids.size();
        current = child;
    }
}

} // namespace tensorlite

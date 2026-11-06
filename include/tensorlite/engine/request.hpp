#pragma once

#include <string>
#include <vector>
#include <optional>
#include <chrono>
#include <functional>
#include "tensorlite/core/types.hpp"

namespace tensorlite {

// Hyper-parameters controlling generation
struct SamplingParams {
    float temperature{1.0f}; // 0.0 = greedy argmax
    float top_p{1.0f};       // Nucleus sampling threshold
    int32_t max_tokens{64};  // Maximum generation limit
    bool ignore_eos{false};
    std::vector<TokenId> stop_token_ids;

    [[nodiscard]] bool is_greedy() const noexcept {
        return temperature <= 1e-4f;
    }
};

// Lifecycle state of a sequence inside the inference engine
enum class SequenceStatus {
    WAITING,        // In waiting queue, awaiting initial block allocation
    RUNNING,        // Active in continuous batching iteration
    FINISHED_STOP,  // Reached EOS or stop token
    FINISHED_LENGTH // Reached max_tokens limit
};

[[nodiscard]] constexpr const char* sequence_status_to_string(SequenceStatus status) noexcept {
    switch (status) {
        case SequenceStatus::WAITING: return "WAITING";
        case SequenceStatus::RUNNING: return "RUNNING";
        case SequenceStatus::FINISHED_STOP: return "FINISHED_STOP";
        case SequenceStatus::FINISHED_LENGTH: return "FINISHED_LENGTH";
    }
    return "UNKNOWN";
}

// Sequence represents a single generation thread (e.g. beam or sample)
class Sequence {
public:
    Sequence(SequenceId id, std::vector<TokenId> prompt_tokens)
        : id_(id),
          prompt_token_count_(prompt_tokens.size()),
          tokens_(std::move(prompt_tokens)),
          status_(SequenceStatus::WAITING) {}

    [[nodiscard]] SequenceId id() const noexcept { return id_; }
    [[nodiscard]] SequenceStatus status() const noexcept { return status_; }
    void set_status(SequenceStatus s) noexcept { status_ = s; }

    [[nodiscard]] size_t total_len() const noexcept { return tokens_.size(); }
    [[nodiscard]] size_t prompt_len() const noexcept { return prompt_token_count_; }
    [[nodiscard]] size_t output_len() const noexcept { return tokens_.size() - prompt_token_count_; }

    [[nodiscard]] size_t num_computed_tokens() const noexcept { return num_computed_tokens_; }
    void set_num_computed_tokens(size_t n) noexcept { num_computed_tokens_ = n; }

    [[nodiscard]] const std::vector<TokenId>& tokens() const noexcept { return tokens_; }
    [[nodiscard]] TokenId last_token() const {
        if (tokens_.empty()) return INVALID_TOKEN_ID;
        return tokens_.back();
    }

    void append_token(TokenId token) {
        tokens_.push_back(token);
    }

    // PagedAttention Block Table: maps sequence logical block index -> engine physical BlockId
    [[nodiscard]] std::vector<BlockId>& block_table() noexcept { return block_table_; }
    [[nodiscard]] const std::vector<BlockId>& block_table() const noexcept { return block_table_; }

    [[nodiscard]] bool is_finished() const noexcept {
        return status_ == SequenceStatus::FINISHED_STOP || status_ == SequenceStatus::FINISHED_LENGTH;
    }

    [[nodiscard]] bool is_prefill() const noexcept {
        return num_computed_tokens_ < prompt_token_count_;
    }

private:
    SequenceId id_;
    size_t prompt_token_count_{0};
    size_t num_computed_tokens_{0};
    std::vector<TokenId> tokens_;
    std::vector<BlockId> block_table_;
    SequenceStatus status_{SequenceStatus::WAITING};
};

// Callback type for streaming tokens back to callers
using TokenStreamCallback = std::function<void(const std::string& token_str, bool is_finished)>;

// User-level Request wrapping one or more sequences
struct Request {
    RequestId id;
    std::string prompt;
    SamplingParams sampling_params;
    std::shared_ptr<Sequence> sequence;
    TokenStreamCallback stream_callback{nullptr};
    std::chrono::steady_clock::time_point arrival_time{std::chrono::steady_clock::now()};
};

} // namespace tensorlite

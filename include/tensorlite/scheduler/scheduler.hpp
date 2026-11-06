#pragma once

#include <vector>
#include <deque>
#include <memory>
#include <algorithm>
#include "tensorlite/engine/request.hpp"
#include "tensorlite/cache/kv_cache_manager.hpp"

namespace tensorlite {

// Scheduler configuration parameters
struct SchedulerConfig {
    int32_t max_num_seqs{16};          // Maximum concurrent sequences in running batch
    int32_t max_num_batched_tokens{512}; // Maximum total tokens processed in one prefill step (Chunked Prefill)
    bool enable_chunked_prefill{true};
};

// Iteration schedule plan returned by the scheduler
struct ScheduleOutput {
    std::vector<std::shared_ptr<Sequence>> prefill_seqs;
    std::vector<int32_t> prefill_chunk_sizes; // Tokens to compute in this step per prefill sequence
    std::vector<std::shared_ptr<Sequence>> decode_seqs;
    std::vector<std::shared_ptr<Sequence>> finished_seqs;

    [[nodiscard]] bool empty() const noexcept {
        return prefill_seqs.empty() && decode_seqs.empty();
    }
};

// Continuous Batching Iteration-level Scheduler
class Scheduler {
public:
    Scheduler(SchedulerConfig config, std::shared_ptr<KVCacheManager> kv_manager)
        : config_(config), kv_manager_(std::move(kv_manager)) {}

    // Add incoming request to the waiting queue
    void add_request(std::shared_ptr<Request> request) {
        waiting_queue_.push_back(request->sequence);
        active_requests_[request->sequence->id()] = request;
    }

    // Main scheduling step executed every engine iteration
    ScheduleOutput schedule();

    // Notify scheduler of finished sequences to free blocks and invoke completion callbacks
    void finish_sequence(SequenceId seq_id, SequenceStatus status);

    [[nodiscard]] size_t num_waiting() const noexcept { return waiting_queue_.size(); }
    [[nodiscard]] size_t num_running() const noexcept { return running_queue_.size(); }
    [[nodiscard]] bool has_work() const noexcept {
        return !waiting_queue_.empty() || !running_queue_.empty();
    }

    [[nodiscard]] std::shared_ptr<Request> get_request(SequenceId seq_id) const {
        auto it = active_requests_.find(seq_id);
        if (it != active_requests_.end()) return it->second;
        return nullptr;
    }

private:
    SchedulerConfig config_;
    std::shared_ptr<KVCacheManager> kv_manager_;

    std::deque<std::shared_ptr<Sequence>> waiting_queue_;
    std::vector<std::shared_ptr<Sequence>> running_queue_;
    std::unordered_map<SequenceId, std::shared_ptr<Request>> active_requests_;
};

} // namespace tensorlite

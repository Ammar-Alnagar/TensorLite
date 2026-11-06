#include "tensorlite/scheduler/scheduler.hpp"

namespace tensorlite {

ScheduleOutput Scheduler::schedule() {
    ScheduleOutput output;

    // 1. Process Running Queue (Decode Sequences)
    // First, verify existing running sequences can allocate their next block if they crossed a block boundary
    std::vector<std::shared_ptr<Sequence>> still_running;
    still_running.reserve(running_queue_.size());

    for (auto& seq : running_queue_) {
        if (seq->is_finished()) {
            output.finished_seqs.push_back(seq);
            continue;
        }

        // Check if next token requires an additional physical block
        size_t current_len = seq->total_len();
        size_t block_size = kv_manager_->config().block_size;
        size_t needed_blocks = (current_len + block_size - 1) / block_size;

        if (needed_blocks > seq->block_table().size()) {
            if (!kv_manager_->can_allocate(1)) {
                // Preemption: Not enough memory -> suspend or wait.
                // In production, we preempt lowest priority sequence back to waiting queue
                continue;
            }
            kv_manager_->allocate_for_sequence(*seq);
        }

        still_running.push_back(seq);
        output.decode_seqs.push_back(seq);
    }
    running_queue_ = std::move(still_running);

    // 2. Admit Sequences from Waiting Queue (Prefill / Chunked Prefill)
    int32_t remaining_token_budget = config_.max_num_batched_tokens;

    while (!waiting_queue_.empty() &&
           running_queue_.size() < static_cast<size_t>(config_.max_num_seqs) &&
           remaining_token_budget > 0) {

        auto seq = waiting_queue_.front();
        size_t prompt_len = seq->prompt_len();
        size_t computed = seq->num_computed_tokens();
        size_t remaining_tokens = prompt_len - computed;

        size_t chunk_size = remaining_tokens;
        if (config_.enable_chunked_prefill && chunk_size > static_cast<size_t>(remaining_token_budget)) {
            chunk_size = remaining_token_budget;
        }

        if (chunk_size == 0) break;

        // Check memory requirement for this sequence's blocks
        size_t total_tokens_after_chunk = computed + chunk_size;
        size_t block_size = kv_manager_->config().block_size;
        size_t needed_blocks = (total_tokens_after_chunk + block_size - 1) / block_size;
        size_t current_blocks = seq->block_table().size();

        if (needed_blocks > current_blocks) {
            size_t blocks_to_alloc = needed_blocks - current_blocks;
            if (!kv_manager_->can_allocate(blocks_to_alloc)) {
                // Cannot admit this sequence right now; wait for running sequences to finish
                break;
            }
            kv_manager_->allocate_for_sequence(*seq);
        }

        waiting_queue_.pop_front();
        output.prefill_seqs.push_back(seq);
        output.prefill_chunk_sizes.push_back(static_cast<int32_t>(chunk_size));
        remaining_token_budget -= static_cast<int32_t>(chunk_size);

        if (computed + chunk_size >= prompt_len) {
            // Entire prefill completed in this step -> sequence transitions to running queue for decoding
            seq->set_status(SequenceStatus::RUNNING);
            running_queue_.push_back(seq);
        } else {
            // Chunked prefill not complete -> push to front of waiting queue to resume next iteration
            waiting_queue_.push_front(seq);
            break; // Stop scheduling further sequences to respect chunk order
        }
    }

    return output;
}

void Scheduler::finish_sequence(SequenceId seq_id, SequenceStatus status) {
    // 1. Find sequence in running queue and mark finished
    auto it = std::find_if(running_queue_.begin(), running_queue_.end(),
                           [seq_id](const auto& s) { return s->id() == seq_id; });
    if (it != running_queue_.end()) {
        (*it)->set_status(status);
        kv_manager_->free_sequence(**it);
        running_queue_.erase(it);
    }

    // 2. Erase active request tracking
    auto req_it = active_requests_.find(seq_id);
    if (req_it != active_requests_.end()) {
        active_requests_.erase(req_it);
    }
}

} // namespace tensorlite

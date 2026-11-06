#include "tensorlite/engine/engine.hpp"
#include <iostream>

namespace tensorlite {

InferenceEngine::InferenceEngine(
    ModelWeights weights,
    CacheConfig cache_config,
    SchedulerConfig sched_config
) : kv_manager_(std::make_shared<KVCacheManager>(cache_config)),
    prefix_cache_(std::make_shared<PrefixCache>(kv_manager_)),
    scheduler_(std::make_shared<Scheduler>(sched_config, kv_manager_)),
    runner_(std::make_unique<ModelRunner>(std::move(weights))) {
}

void InferenceEngine::add_request(
    const std::string& request_id,
    const std::string& prompt,
    const SamplingParams& params,
    TokenStreamCallback callback
) {
    auto req = std::make_shared<Request>();
    req->id = RequestId{request_id};
    req->prompt = prompt;
    req->sampling_params = params;
    req->stream_callback = std::move(callback);

    auto prompt_tokens = tokenizer_.encode(prompt);
    int64_t sid = ++sequence_counter_;
    req->sequence = std::make_shared<Sequence>(SequenceId{sid}, prompt_tokens);

    // Check prefix cache for reusable blocks!
    auto match = prefix_cache_->match_prefix(prompt_tokens);
    if (match.matched_tokens > 0) {
        req->sequence->block_table() = match.reusable_blocks;
        req->sequence->set_num_computed_tokens(match.matched_tokens);
    }

    scheduler_->add_request(req);
}

bool InferenceEngine::step() {
    if (!scheduler_->has_work()) {
        return false;
    }

    // 1. Invoke scheduler to plan this iteration
    ScheduleOutput plan = scheduler_->schedule();
    if (plan.empty()) {
        return scheduler_->has_work();
    }

    // 2. Execute Prefill sequences (if any)
    for (size_t i = 0; i < plan.prefill_seqs.size(); ++i) {
        auto seq = plan.prefill_seqs[i];
        int32_t chunk_size = plan.prefill_chunk_sizes[i];
        int32_t start_pos = static_cast<int32_t>(seq->num_computed_tokens());

        auto logits = runner_->forward_prefill(*seq, *kv_manager_, start_pos, chunk_size);
        seq->set_num_computed_tokens(start_pos + chunk_size);

        // If sequence finished its entire prompt prefill, sample the first generated token!
        if (seq->num_computed_tokens() >= seq->prompt_len()) {
            auto req = scheduler_->get_request(seq->id());
            assert(req != nullptr);

            TokenId next_token = sampler_.sample(logits, req->sampling_params);
            handle_sampled_token(*seq, next_token, *req);

            // Register full prompt blocks into prefix cache
            prefix_cache_->insert(seq->tokens(), seq->block_table());
        }
    }

    // 3. Execute Continuous Batch Decode (PagedAttention)
    if (!plan.decode_seqs.empty()) {
        auto batch_logits = runner_->forward_decode_batch(plan.decode_seqs, *kv_manager_);

        for (size_t s = 0; s < plan.decode_seqs.size(); ++s) {
            auto seq = plan.decode_seqs[s];
            auto req = scheduler_->get_request(seq->id());
            assert(req != nullptr);

            TokenId next_token = sampler_.sample(batch_logits[s], req->sampling_params);
            handle_sampled_token(*seq, next_token, *req);
        }
    }

    return scheduler_->has_work();
}

void InferenceEngine::handle_sampled_token(Sequence& seq, TokenId next_token, const Request& req) {
    seq.append_token(next_token);
    std::string token_str = tokenizer_.decode(next_token);

    // Check stop criteria
    bool is_eos = (next_token == tokenizer_.eos_token_id());
    bool hit_stop_token = false;
    for (auto st : req.sampling_params.stop_token_ids) {
        if (next_token == st) {
            hit_stop_token = true;
            break;
        }
    }

    bool length_limit = (seq.output_len() >= static_cast<size_t>(req.sampling_params.max_tokens));

    bool finished = (is_eos || hit_stop_token || length_limit);

    // Stream token string to user callback
    if (req.stream_callback) {
        req.stream_callback(token_str, finished);
    }

    if (finished) {
        SequenceStatus status = (is_eos || hit_stop_token) ? SequenceStatus::FINISHED_STOP : SequenceStatus::FINISHED_LENGTH;
        scheduler_->finish_sequence(seq.id(), status);
    }
}

} // namespace tensorlite

#include <cassert>
#include <iostream>
#include <memory>
#include <vector>
#include "tensorlite/scheduler/scheduler.hpp"

using namespace tensorlite;

void test_continuous_batching_scheduling() {
    CacheConfig cache_cfg;
    cache_cfg.block_size = 16;
    cache_cfg.num_blocks = 10; // 10 physical blocks total
    auto kv_mgr = std::make_shared<KVCacheManager>(cache_cfg);

    SchedulerConfig sched_cfg;
    sched_cfg.max_num_seqs = 4;
    sched_cfg.max_num_batched_tokens = 32;
    sched_cfg.enable_chunked_prefill = true;

    Scheduler scheduler(sched_cfg, kv_mgr);

    // Request 1: 20 tokens -> needs 2 blocks
    auto req1 = std::make_shared<Request>();
    req1->id = RequestId{"r1"};
    req1->sequence = std::make_shared<Sequence>(SequenceId{1}, std::vector<TokenId>(20, TokenId{1}));
    scheduler.add_request(req1);

    // Request 2: 10 tokens -> needs 1 block
    auto req2 = std::make_shared<Request>();
    req2->id = RequestId{"r2"};
    req2->sequence = std::make_shared<Sequence>(SequenceId{2}, std::vector<TokenId>(10, TokenId{2}));
    scheduler.add_request(req2);

    assert(scheduler.num_waiting() == 2);
    assert(scheduler.num_running() == 0);

    // Iteration 1 schedule:
    // Token budget = 32. Req1 has 20 tokens (takes 20), Req2 has 10 tokens (takes 10). Total = 30 <= 32.
    // Both req1 and req2 complete prefill and enter running queue!
    auto plan1 = scheduler.schedule();
    assert(plan1.prefill_seqs.size() == 2);
    assert(plan1.prefill_chunk_sizes[0] == 20);
    assert(plan1.prefill_chunk_sizes[1] == 10);
    assert(scheduler.num_running() == 2);
    assert(scheduler.num_waiting() == 0);

    // Iteration 2 schedule:
    // Both seqs are now running and decode 1 token each
    auto plan2 = scheduler.schedule();
    assert(plan2.prefill_seqs.empty());
    assert(plan2.decode_seqs.size() == 2);

    // Finish seq 1
    scheduler.finish_sequence(SequenceId{1}, SequenceStatus::FINISHED_STOP);
    assert(scheduler.num_running() == 1);

    // Iteration 3: only seq 2 remains
    auto plan3 = scheduler.schedule();
    assert(plan3.decode_seqs.size() == 1);
    assert(plan3.decode_seqs[0]->id() == SequenceId{2});

    std::cout << "[PASS] Continuous batching iteration schedule, prefill admission, and finish verified.\n";
}

void test_chunked_prefill() {
    CacheConfig cache_cfg;
    cache_cfg.block_size = 16;
    cache_cfg.num_blocks = 10;
    auto kv_mgr = std::make_shared<KVCacheManager>(cache_cfg);

    SchedulerConfig sched_cfg;
    sched_cfg.max_num_seqs = 4;
    sched_cfg.max_num_batched_tokens = 25; // Small budget to enforce chunking
    sched_cfg.enable_chunked_prefill = true;

    Scheduler scheduler(sched_cfg, kv_mgr);

    // Sequence with 50 tokens (exceeds budget 25)
    auto req = std::make_shared<Request>();
    req->id = RequestId{"long_req"};
    req->sequence = std::make_shared<Sequence>(SequenceId{10}, std::vector<TokenId>(50, TokenId{5}));
    scheduler.add_request(req);

    // Step 1: Takes first chunk of 25 tokens
    auto plan1 = scheduler.schedule();
    assert(plan1.prefill_seqs.size() == 1);
    assert(plan1.prefill_chunk_sizes[0] == 25);
    assert(scheduler.num_running() == 0);
    assert(scheduler.num_waiting() == 1);

    // Simulate prefill chunk completed on sequence
    req->sequence->set_num_computed_tokens(25);

    // Step 2: Takes remaining 25 tokens
    auto plan2 = scheduler.schedule();
    assert(plan2.prefill_seqs.size() == 1);
    assert(plan2.prefill_chunk_sizes[0] == 25);
    assert(scheduler.num_running() == 1); // Now admitted to running!
    assert(scheduler.num_waiting() == 0);

    std::cout << "[PASS] Chunked prefill token-budget splitting verified.\n";
}

int main() {
    std::cout << "Running Module 7 Tests...\n";
    test_continuous_batching_scheduling();
    test_chunked_prefill();
    std::cout << "All Module 7 tests completed successfully!\n";
    return 0;
}

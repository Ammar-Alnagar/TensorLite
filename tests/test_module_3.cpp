#include <cassert>
#include <iostream>
#include <memory>
#include "tensorlite/engine/tokenizer.hpp"
#include "tensorlite/engine/request.hpp"

using namespace tensorlite;

void test_tokenizer_encode_decode() {
    Tokenizer tokenizer;
    tokenizer.add_special_token("<|im_start|>", TokenId{151644});
    tokenizer.add_special_token("<|im_end|>", TokenId{151645});

    std::string text = "Hello <|im_start|>World<|im_end|>";
    auto encoded = tokenizer.encode(text);
    assert(!encoded.empty());

    // Check special token preservation
    bool found_start = false;
    bool found_end = false;
    for (auto t : encoded) {
        if (t.get() == 151644) found_start = true;
        if (t.get() == 151645) found_end = true;
    }
    assert(found_start);
    assert(found_end);

    std::string decoded = tokenizer.decode(encoded);
    assert(decoded == text);
    std::cout << "[PASS] Tokenizer byte-fallback and special tokens verified.\n";
}

void test_sequence_lifecycle() {
    std::vector<TokenId> prompt = {TokenId{10}, TokenId{20}, TokenId{30}};
    Sequence seq(SequenceId{1}, prompt);

    assert(seq.prompt_len() == 3);
    assert(seq.total_len() == 3);
    assert(seq.output_len() == 0);
    assert(seq.is_prefill());
    assert(seq.status() == SequenceStatus::WAITING);

    // Transition to RUNNING
    seq.set_status(SequenceStatus::RUNNING);
    seq.set_num_computed_tokens(3);
    assert(!seq.is_prefill());

    // Generate token
    seq.append_token(TokenId{40});
    assert(seq.total_len() == 4);
    assert(seq.output_len() == 1);
    assert(seq.last_token().get() == 40);

    // Block table binding
    seq.block_table().push_back(BlockId{101});
    assert(seq.block_table().size() == 1);
    assert(seq.block_table()[0].get() == 101);

    std::cout << "[PASS] Sequence state and token transitions verified.\n";
}

void test_request_and_callbacks() {
    Request req;
    req.id = RequestId{"req-001"};
    req.prompt = "Write a quick test";
    req.sampling_params.max_tokens = 16;
    req.sampling_params.temperature = 0.0f; // Greedy

    std::string streamed;
    req.stream_callback = [&](const std::string& token, bool is_finished) {
        streamed += token;
    };

    req.stream_callback("Hello ", false);
    req.stream_callback("world!", true);
    assert(streamed == "Hello world!");

    std::cout << "[PASS] Request sampling params and streaming callback verified.\n";
}

int main() {
    std::cout << "Running Module 3 Tests...\n";
    test_tokenizer_encode_decode();
    test_sequence_lifecycle();
    test_request_and_callbacks();
    std::cout << "All Module 3 tests completed successfully!\n";
    return 0;
}

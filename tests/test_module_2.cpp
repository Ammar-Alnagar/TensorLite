#include <cassert>
#include <iostream>
#include <filesystem>
#include "tensorlite/model/model_config.hpp"
#include "tensorlite/model/safetensors_loader.hpp"
#include "tensorlite/model/model_weights.hpp"

using namespace tensorlite;

void test_model_config() {
    std::filesystem::path cfg_path = "models/test_qwen_fp8/config.json";
    auto config = ModelConfig::from_json_file(cfg_path);

    assert(config.model_type == "qwen2");
    assert(config.vocab_size == 256);
    assert(config.hidden_size == 64);
    assert(config.intermediate_size == 128);
    assert(config.num_hidden_layers == 2);
    assert(config.num_attention_heads == 4);
    assert(config.num_key_value_heads == 2);
    assert(config.head_dim == 16);
    assert(config.quantization == "fp8");

    std::cout << "[PASS] ModelConfig JSON parsing verified.\n";
}

void test_safetensors_loader() {
    std::filesystem::path st_path = "models/test_qwen_fp8/model.safetensors";
    SafetensorsLoader loader(st_path);

    assert(loader.contains("model.embed_tokens.weight"));
    assert(loader.contains("model.layers.0.self_attn.q_proj.weight"));

    auto embed = loader.load_tensor("model.embed_tokens.weight");
    assert(embed.dtype == DataType::FP32);
    assert(embed.shape.size() == 2);
    assert(embed.shape[0] == 256);
    assert(embed.shape[1] == 64);
    assert(embed.data.size_bytes() == 256 * 64 * sizeof(float));

    auto q_proj = loader.load_tensor("model.layers.0.self_attn.q_proj.weight");
    assert(q_proj.dtype == DataType::FP8_E4M3);
    assert(q_proj.shape.size() == 2);
    assert(q_proj.shape[0] == 64);
    assert(q_proj.shape[1] == 64);
    assert(q_proj.data.size_bytes() == 64 * 64 * 1);
    assert(q_proj.scale == 0.01f);

    std::cout << "[PASS] SafetensorsLoader tensor extraction & scale loading verified.\n";
}

void test_model_weights_mapping() {
    auto weights = ModelWeights::load_from_hf_dir("models/test_qwen_fp8");
    assert(weights.layers.size() == 2);
    assert(weights.embed_tokens.shape[0] == 256);
    assert(weights.layers[0].q_proj.shape[0] == 64);
    assert(weights.layers[0].k_proj.shape[0] == 32);
    assert(weights.layers[0].v_proj.shape[0] == 32);
    assert(weights.layers[0].gate_proj.shape[0] == 128);

    std::cout << "[PASS] ModelWeights full HF directory mapping verified.\n";
}

int main() {
    std::cout << "Running Module 2 Tests...\n";
    test_model_config();
    test_safetensors_loader();
    test_model_weights_mapping();
    std::cout << "All Module 2 tests completed successfully!\n";
    return 0;
}

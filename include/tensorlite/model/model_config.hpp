#pragma once

#include <string>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include "tensorlite/core/error.hpp"

namespace tensorlite {

// Model hyper-parameters matching modern decoder-only architectures (Qwen / LLaMA)
struct ModelConfig {
    std::string model_type{"qwen"};
    int32_t vocab_size{151936};
    int32_t hidden_size{896};
    int32_t intermediate_size{4864};
    int32_t num_hidden_layers{24};
    int32_t num_attention_heads{14};
    int32_t num_key_value_heads{2}; // Grouped Query Attention (GQA)
    int32_t head_dim{64};            // hidden_size / num_attention_heads
    float rms_norm_eps{1e-6f};
    float rope_theta{1000000.0f};
    int32_t max_position_embeddings{32768};
    bool tie_word_embeddings{true};
    std::string quantization{"fp8"};

    static ModelConfig from_json_file(const std::filesystem::path& config_path) {
        if (!std::filesystem::exists(config_path)) {
            throw ModelLoadException(std::format("Config file does not exist: {}", config_path.string()));
        }

        std::ifstream file(config_path);
        if (!file.is_open()) {
            throw ModelLoadException(std::format("Failed to open config file: {}", config_path.string()));
        }

        nlohmann::json j;
        try {
            file >> j;
        } catch (const std::exception& e) {
            throw ModelLoadException(std::format("Failed to parse JSON config: {}", e.what()));
        }

        ModelConfig cfg;
        if (j.contains("model_type")) cfg.model_type = j["model_type"].get<std::string>();
        if (j.contains("vocab_size")) cfg.vocab_size = j["vocab_size"].get<int32_t>();
        if (j.contains("hidden_size")) cfg.hidden_size = j["hidden_size"].get<int32_t>();
        if (j.contains("intermediate_size")) cfg.intermediate_size = j["intermediate_size"].get<int32_t>();
        if (j.contains("num_hidden_layers")) cfg.num_hidden_layers = j["num_hidden_layers"].get<int32_t>();
        if (j.contains("num_attention_heads")) cfg.num_attention_heads = j["num_attention_heads"].get<int32_t>();

        if (j.contains("num_key_value_heads")) {
            cfg.num_key_value_heads = j["num_key_value_heads"].get<int32_t>();
        } else {
            cfg.num_key_value_heads = cfg.num_attention_heads; // Standard MHA fallback
        }

        if (j.contains("head_dim")) {
            cfg.head_dim = j["head_dim"].get<int32_t>();
        } else {
            cfg.head_dim = cfg.hidden_size / cfg.num_attention_heads;
        }

        if (j.contains("rms_norm_eps")) cfg.rms_norm_eps = j["rms_norm_eps"].get<float>();
        if (j.contains("rope_theta")) cfg.rope_theta = j["rope_theta"].get<float>();
        if (j.contains("max_position_embeddings")) cfg.max_position_embeddings = j["max_position_embeddings"].get<int32_t>();
        if (j.contains("tie_word_embeddings")) cfg.tie_word_embeddings = j["tie_word_embeddings"].get<bool>();

        if (j.contains("quantization_config")) {
            cfg.quantization = "fp8";
        }

        return cfg;
    }
};

} // namespace tensorlite

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include "tensorlite/model/model_config.hpp"
#include "tensorlite/model/safetensors_loader.hpp"

namespace tensorlite {

// Layer-level weights for Qwen/LLaMA decoder block
struct TransformerLayerWeights {
    // Attention
    WeightTensor input_layernorm;
    WeightTensor q_proj;
    WeightTensor k_proj;
    WeightTensor v_proj;
    WeightTensor o_proj;

    // MLP
    WeightTensor post_attention_layernorm;
    WeightTensor gate_proj;
    WeightTensor up_proj;
    WeightTensor down_proj;
};

// Top-level model weights loaded in memory
struct ModelWeights {
    ModelConfig config;
    WeightTensor embed_tokens;
    std::vector<TransformerLayerWeights> layers;
    WeightTensor final_norm;
    WeightTensor lm_head;

    static ModelWeights load_from_hf_dir(const std::filesystem::path& model_dir) {
        ModelWeights weights;
        weights.config = ModelConfig::from_json_file(model_dir / "config.json");

        std::filesystem::path st_path = model_dir / "model.safetensors";
        SafetensorsLoader loader(st_path);

        std::unordered_map<std::string, WeightTensor> all_weights;
        loader.load_all(all_weights);

        // Map embeddings
        if (all_weights.contains("model.embed_tokens.weight")) {
            weights.embed_tokens = std::move(all_weights["model.embed_tokens.weight"]);
        } else {
            throw ModelLoadException("Missing model.embed_tokens.weight");
        }

        // Map layers
        weights.layers.resize(weights.config.num_hidden_layers);
        for (int32_t i = 0; i < weights.config.num_hidden_layers; ++i) {
            std::string pfx = std::format("model.layers.{}", i);
            auto& l = weights.layers[i];

            l.input_layernorm = std::move(all_weights[pfx + ".input_layernorm.weight"]);
            l.q_proj = std::move(all_weights[pfx + ".self_attn.q_proj.weight"]);
            l.k_proj = std::move(all_weights[pfx + ".self_attn.k_proj.weight"]);
            l.v_proj = std::move(all_weights[pfx + ".self_attn.v_proj.weight"]);
            l.o_proj = std::move(all_weights[pfx + ".self_attn.o_proj.weight"]);

            l.post_attention_layernorm = std::move(all_weights[pfx + ".post_attention_layernorm.weight"]);
            l.gate_proj = std::move(all_weights[pfx + ".mlp.gate_proj.weight"]);
            l.up_proj = std::move(all_weights[pfx + ".mlp.up_proj.weight"]);
            l.down_proj = std::move(all_weights[pfx + ".mlp.down_proj.weight"]);
        }

        // Map final norm and lm_head
        weights.final_norm = std::move(all_weights["model.norm.weight"]);
        if (all_weights.contains("lm_head.weight")) {
            weights.lm_head = std::move(all_weights["lm_head.weight"]);
        }

        return weights;
    }
};

} // namespace tensorlite

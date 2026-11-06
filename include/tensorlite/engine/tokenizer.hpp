#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <memory>
#include <nlohmann/json.hpp>
#include "tensorlite/core/types.hpp"
#include "tensorlite/core/error.hpp"

namespace tensorlite {

// Lightweight, self-contained BPE / Byte-level Tokenizer
// Compatible with Qwen / LLaMA vocabulary files
class Tokenizer {
public:
    Tokenizer() {
        // Fallback default ASCII vocabulary so the tokenizer is immediately usable
        for (int i = 0; i < 256; ++i) {
            std::string s(1, static_cast<char>(i));
            token_to_id_[s] = TokenId{i};
            id_to_token_[TokenId{i}] = s;
        }
    }

    void add_special_token(const std::string& token, TokenId id) {
        token_to_id_[token] = id;
        id_to_token_[id] = token;
        if (token == "<|endoftext|>" || token == "<|im_end|>" || token == "</s>") {
            eos_token_id_ = id;
        }
        if (token == "<|im_start|>" || token == "<s>") {
            bos_token_id_ = id;
        }
    }

    void load_vocab_json(const std::filesystem::path& vocab_path) {
        if (!std::filesystem::exists(vocab_path)) {
            throw ModelLoadException(std::format("Vocab file not found: {}", vocab_path.string()));
        }

        std::ifstream f(vocab_path);
        nlohmann::json j;
        f >> j;

        token_to_id_.clear();
        id_to_token_.clear();

        for (auto it = j.begin(); it != j.end(); ++it) {
            TokenId tid{it.value().get<int32_t>()};
            token_to_id_[it.key()] = tid;
            id_to_token_[tid] = it.key();
        }
    }

    [[nodiscard]] std::vector<TokenId> encode(const std::string& text) const {
        std::vector<TokenId> tokens;
        tokens.reserve(text.size());

        // Byte-fallback encoding: for each character or byte, look up or fallback to raw byte
        size_t i = 0;
        while (i < text.size()) {
            // Check for multi-character special tokens first (e.g. <|im_start|>)
            bool matched = false;
            if (text[i] == '<') {
                for (const auto& [token, id] : token_to_id_) {
                    if (token.size() > 1 && text.compare(i, token.size(), token) == 0) {
                        tokens.push_back(id);
                        i += token.size();
                        matched = true;
                        break;
                    }
                }
            }

            if (!matched) {
                std::string ch(1, text[i]);
                auto it = token_to_id_.find(ch);
                if (it != token_to_id_.end()) {
                    tokens.push_back(it->second);
                } else {
                    tokens.push_back(TokenId{static_cast<uint8_t>(text[i])});
                }
                i++;
            }
        }
        return tokens;
    }

    [[nodiscard]] std::string decode(TokenId id) const {
        auto it = id_to_token_.find(id);
        if (it != id_to_token_.end()) {
            return it->second;
        }
        if (id.get() >= 0 && id.get() < 256) {
            return std::string(1, static_cast<char>(id.get()));
        }
        return std::format("<token_{}>", id.get());
    }

    [[nodiscard]] std::string decode(const std::vector<TokenId>& tokens) const {
        std::string res;
        for (auto tid : tokens) {
            res += decode(tid);
        }
        return res;
    }

    [[nodiscard]] TokenId eos_token_id() const noexcept { return eos_token_id_; }
    [[nodiscard]] TokenId bos_token_id() const noexcept { return bos_token_id_; }

private:
    std::unordered_map<std::string, TokenId> token_to_id_;
    std::unordered_map<TokenId, std::string> id_to_token_;
    TokenId eos_token_id_{TokenId{151643}}; // Default Qwen2/3 <|im_end|> or <|endoftext|>
    TokenId bos_token_id_{TokenId{151644}};
};

} // namespace tensorlite

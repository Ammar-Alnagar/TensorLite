#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include "tensorlite/core/types.hpp"
#include "tensorlite/engine/request.hpp"

namespace tensorlite {

// Token Sampler supporting Greedy Argmax and Multinomial Top-P Nucleus sampling
class Sampler {
public:
    explicit Sampler(uint64_t seed = 42) : rng_(seed) {}

    TokenId sample(const std::vector<float>& logits, const SamplingParams& params) {
        if (logits.empty()) return INVALID_TOKEN_ID;

        // 1. Greedy Argmax
        if (params.is_greedy()) {
            auto max_it = std::max_element(logits.begin(), logits.end());
            int32_t best_idx = static_cast<int32_t>(std::distance(logits.begin(), max_it));
            return TokenId{best_idx};
        }

        // 2. Temperature scaling
        float temp = std::max(1e-4f, params.temperature);
        std::vector<float> scaled_logits(logits.size());
        float max_l = -1e30f;
        for (size_t i = 0; i < logits.size(); ++i) {
            scaled_logits[i] = logits[i] / temp;
            if (scaled_logits[i] > max_l) max_l = scaled_logits[i];
        }

        // Softmax
        std::vector<float> probs(logits.size());
        float sum_exp = 0.0f;
        for (size_t i = 0; i < logits.size(); ++i) {
            probs[i] = std::exp(scaled_logits[i] - max_l);
            sum_exp += probs[i];
        }
        float inv_sum = 1.0f / sum_exp;
        for (float& p : probs) p *= inv_sum;

        // 3. Top-P (Nucleus) filter
        if (params.top_p < 1.0f) {
            std::vector<int32_t> indices(probs.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::sort(indices.begin(), indices.end(), [&](int a, int b) {
                return probs[a] > probs[b];
            });

            float cumsum = 0.0f;
            float cutoff_sum = 0.0f;
            std::vector<std::pair<int32_t, float>> filtered;

            for (int32_t idx : indices) {
                cumsum += probs[idx];
                filtered.emplace_back(idx, probs[idx]);
                cutoff_sum += probs[idx];
                if (cumsum >= params.top_p) break;
            }

            // Sample from filtered distribution
            std::uniform_real_distribution<float> dist(0.0f, cutoff_sum);
            float r = dist(rng_);
            float cur = 0.0f;
            for (const auto& [idx, p] : filtered) {
                cur += p;
                if (cur >= r) return TokenId{idx};
            }
            return TokenId{filtered.back().first};
        }

        // Multinomial sample
        std::discrete_distribution<int32_t> dist(probs.begin(), probs.end());
        return TokenId{dist(rng_)};
    }

private:
    std::mt19937_64 rng_;
};

} // namespace tensorlite

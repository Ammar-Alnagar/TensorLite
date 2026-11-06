#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include "tensorlite/core/types.hpp"
#include "tensorlite/model/safetensors_loader.hpp"

namespace tensorlite {

// Fundamental transformer operators
class Ops {
public:
    // Root Mean Square Normalization (RMSNorm)
    // y = (x / sqrt(mean(x^2) + eps)) * weight
    static void rms_norm(
        float* out,
        const float* in,
        const float* weight,
        int32_t rows,
        int32_t hidden_size,
        float eps = 1e-6f
    ) {
        for (int32_t r = 0; r < rows; ++r) {
            const float* in_row = in + r * hidden_size;
            float* out_row = out + r * hidden_size;

            float sum_sq = 0.0f;
            for (int32_t c = 0; c < hidden_size; ++c) {
                sum_sq += in_row[c] * in_row[c];
            }
            float rms = 1.0f / std::sqrt((sum_sq / static_cast<float>(hidden_size)) + eps);

            for (int32_t c = 0; c < hidden_size; ++c) {
                out_row[c] = in_row[c] * rms * weight[c];
            }
        }
    }

    // Rotary Position Embeddings (RoPE) applied to Q and K
    static void apply_rotary_emb(
        float* qk,
        int32_t seq_len,
        int32_t num_heads,
        int32_t head_dim,
        int32_t start_pos,
        float theta = 10000.0f
    ) {
        int32_t half_dim = head_dim / 2;
        for (int32_t i = 0; i < seq_len; ++i) {
            int32_t pos = start_pos + i;
            for (int32_t h = 0; h < num_heads; ++h) {
                float* vec = qk + (i * num_heads + h) * head_dim;
                for (int32_t d = 0; d < half_dim; ++d) {
                    float freq = 1.0f / std::pow(theta, static_cast<float>(2 * d) / static_cast<float>(head_dim));
                    float angle = static_cast<float>(pos) * freq;
                    float cos_val = std::cos(angle);
                    float sin_val = std::sin(angle);

                    float x0 = vec[d];
                    float x1 = vec[d + half_dim];
                    vec[d] = x0 * cos_val - x1 * sin_val;
                    vec[d + half_dim] = x0 * sin_val + x1 * cos_val;
                }
            }
        }
    }

    // Linear Projection supporting FP8 weights and scales
    // Y [M, N] = X [M, K] * (W [N, K])^T * scale
    static void linear(
        float* out,
        const float* x,
        const WeightTensor& w,
        int32_t m,
        int32_t n,
        int32_t k
    ) {
        float scale = w.scale;
        const fp8_e4m3* w_fp8 = static_cast<const fp8_e4m3*>(w.data.raw_ptr());
        const float* w_f32 = static_cast<const float*>(w.data.raw_ptr());
        bool is_fp8 = (w.dtype == DataType::FP8_E4M3);

        for (int32_t r = 0; r < m; ++r) {
            const float* x_row = x + r * k;
            float* out_row = out + r * n;

            for (int32_t c = 0; c < n; ++c) {
                float sum = 0.0f;
                if (is_fp8) {
                    const fp8_e4m3* w_row = w_fp8 + c * k;
                    for (int32_t i = 0; i < k; ++i) {
                        sum += x_row[i] * w_row[i].to_float();
                    }
                    out_row[c] = sum * scale;
                } else {
                    const float* w_row = w_f32 + c * k;
                    for (int32_t i = 0; i < k; ++i) {
                        sum += x_row[i] * w_row[i];
                    }
                    out_row[c] = sum;
                }
            }
        }
    }

    // SwiGLU Activation: SiLU(gate) * up
    // SiLU(x) = x / (1 + exp(-x))
    static void swiglu(
        float* out,
        const float* gate,
        const float* up,
        int32_t num_elements
    ) {
        for (int32_t i = 0; i < num_elements; ++i) {
            float g = gate[i];
            float silu_g = g / (1.0f + std::exp(-g));
            out[i] = silu_g * up[i];
        }
    }

    // Vector addition / residual connection: dst += src
    static void add_residual(float* dst, const float* src, int32_t num_elements) {
        for (int32_t i = 0; i < num_elements; ++i) {
            dst[i] += src[i];
        }
    }
};

} // namespace tensorlite

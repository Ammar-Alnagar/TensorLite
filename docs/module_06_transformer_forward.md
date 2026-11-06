# Module 6: Transformer Layer and Model Forward (Qwen Architecture)

In this module, we construct the end-to-end forward pass of a modern decoder-only transformer (Qwen 2/2.5/3, LLaMA 3/3.1).

## 1. Transformer Architecture Data Flow

```mermaid
graph TD
    In[Input Token IDs] --> Embed[Embedding Lookup Table]
    Embed --> L0[Transformer Layer 0]
    L0 --> L1[Transformer Layer 1]
    L1 --> Norm[Final RMSNorm]
    Norm --> LM[LM Head / Unembedding Projection]
    LM --> Logits[Logits: vocab_size]

    subgraph Layer Computation
        LIn[Hidden States x] --> RMS1[Input RMSNorm]
        RMS1 --> ProjQKV[Q, K, V Linear Projections]
        ProjQKV --> RoPE[Apply RoPE]
        RoPE --> CacheWrite[Write K, V to Paged KV Cache]
        CacheWrite --> Attn[PagedAttention or Causal Prefill]
        Attn --> OutProj[O Proj Linear]
        OutProj --> Res1[Residual Addition: x + attn_out]
        Res1 --> RMS2[Post-Attention RMSNorm]
        RMS2 --> MLP_Gate[Gate Proj & Up Proj]
        MLP_Gate --> SwiGLU[SwiGLU Activation: SiLU(gate) * up]
        SwiGLU --> MLP_Down[Down Proj Linear]
        MLP_Down --> Res2[Residual Addition: x + mlp_out]
    end
```

## 2. Key Components

### 2.1 Rotary Position Embeddings (RoPE)
Instead of adding absolute position embeddings to token vectors, RoPE rotates 2D slices of query and key head vectors by angle $m \theta_i$:
$$ \begin{pmatrix} q_0' \\ q_1' \end{pmatrix} = \begin{pmatrix} \cos(m \theta) & -\sin(m \theta) \\ \sin(m \theta) & \cos(m \theta) \end{pmatrix} \begin{pmatrix} q_0 \\ q_1 \end{pmatrix} $$
This allows the dot-product attention score $\langle q_m, k_n \rangle$ to naturally depend on relative distance $(m - n)$.

### 2.2 SwiGLU Feed-Forward Network
Standard MLPs use ReLU or GELU: $Y = \text{Activation}(X W_1) W_2$.
SwiGLU splits the expansion layer into two matrices:
$$ \text{SwiGLU}(X) = \left( \text{SiLU}(X W_{\text{gate}}) \odot (X W_{\text{up}}) \right) W_{\text{down}} $$
where $\text{SiLU}(x) = x \cdot \sigma(x)$.

### 2.3 Prefill vs. Decode Forward Paths
- **Prefill**: Computes full causal multi-token attention across prompt tokens and stores keys/values into pre-allocated physical cache blocks.
- **Decode**: Executes iteration-level decoding across all active sequences in a single step using the batched `paged_attention_decode` kernel.

## 3. Verification and Tests
Run the test suite:
```bash
ctest -R test_module_6 --output-on-failure
```
Output:
```text
[PASS] Prefill forward pass completed, logits dimension: 256
[PASS] Batch Decode with PagedAttention forward pass completed.
```
In Module 7, we build the engine's core intelligence: **The Continuous Batching Scheduler**!

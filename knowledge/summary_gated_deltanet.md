# Gated Delta Networks: Improving Mamba2 with Delta Rule

**Paper:** Songlin Yang, Jan Kautz, Ali Hatamizadeh (2024). "Gated Delta Networks: Improving Mamba2 with Delta Rule." arXiv:2412.06464, ICLR 2025.

## Core Idea

Gated DeltaNet combines two complementary mechanisms for recurrent state management:
- **Gating (from Mamba2):** Scalar decay α_t ∈ (0,1) for adaptive memory erasure
- **Delta rule (from DeltaNet):** Error-correcting updates for precise key-value association

The **gated delta rule** recurrence:
```
S_t = S_{t-1} * (α_t * (I - β_t * k_t k_t^T)) + β_t * v_t k_t^T
```

Where:
- S_t ∈ R^{d_v × d_k} is the recurrent state (like a fast weight matrix)
- α_t ∈ (0,1) is the gating/decay term (Mamba2-style, from A_log + dt_bias)
- β_t ∈ (0,1) is the delta update strength (sigmoid of linear projection)
- k_t is L2-normalized for stability

## Why It's Better Than Mamba2

| Model | State Update | Weakness |
|-------|-------------|----------|
| Linear Attention | S_t = S_{t-1} + v_t k_t^T | Can only add, never erase → memory overload |
| Mamba2 | S_t = α_t S_{t-1} + v_t k_t^T | Decays everything uniformly → forgets needles in haystack |
| DeltaNet | S_t = S_{t-1}(I - β_t k_t k_t^T) + β_t v_t k_t^T | No decay → memory collision at long sequences |
| **Gated DeltaNet** | S_t = α_t S_{t-1}(I - β_t k_t k_t^T) + β_t v_t k_t^T | **Best of both** |

Key insight: The delta term `(I - β_t k_t k_t^T)` erases the old value associated with key k_t before writing the new one. This is like SGD on the objective `||S_t k_t - v_t||²` — correcting prediction errors.

## Architecture

The GDN block follows Llama's macro design (alternating token mixer + SwiGLU MLP):

```
input → q_proj → conv1d → SiLU → q (L2 norm)
      → k_proj → conv1d → SiLU → k (L2 norm)
      → v_proj → conv1d → SiLU → v
      → a_proj → (with A_log, dt_bias) → g (gating, like Mamba2's α)
      → b_proj → sigmoid → β (delta update strength)
      → g_proj → gate for output

      gated_delta_rule(q, k, v, g, β) → o

      RMSNormGated(o, gate) → o_proj → output
```

Key architectural choices:
- **Separate Q, K, V projections** (not a single in_proj like Mamba2)
- **Short conv1d on Q, K, V separately** (not grouped like Mamba2)
- **L2 normalization on Q, K** inside the kernel for stability
- **GVA (Grouped Value Attention):** num_v_heads can be > num_heads, with expand_v=2
- **Output gate** with gated RMSNorm (like Mamba2)

### Dimensions (from FLA implementation)
```python
key_dim = num_heads * head_dim           # e.g., 6 * 256 = 1536
value_dim = num_v_heads * head_v_dim     # e.g., 6 * 512 = 3072 (with expand_v=2)
# Total ≈ 6 * hidden_size² parameters per layer (same as Mamba2/attention)
```

## Chunked Training Algorithm

Similar to Mamba2's SSD but with the WY representation for Householder products:

1. **Compute WY transform** within each chunk: Converts cumulative products of Householder matrices `∏(I - β_t k_t k_t^T)` into an efficient matrix form using the classical WY representation.

2. **Intra-chunk:** Quadratic attention-like matmul with delta-corrected values:
   ```
   T = [I + strictLower(diag(β) * (Γ ⊙ KK^T))]^{-1} * diag(β)
   U_tilde = T @ V  (delta-corrected values)
   W = T @ K        (delta-corrected keys for state update)
   O_intra = (QK^T ⊙ M) @ (U_tilde - W @ S^T)
   ```

3. **Inter-chunk state propagation:**
   ```
   S_{t+1} = decay(S_t) + (U_tilde - W_decay @ S_t^T)^T @ K_decay
   ```

4. **Output:** `O = Q_decay @ S^T + O_intra`

## Real-World Adoption
- **Qwen 3.5:** Uses GDN layers (3:1 ratio with full attention)
- **OLMo Hybrid:** Incorporates GDN
- Consistently outperforms Mamba2 on retrieval, reasoning, and long-context tasks

## Connection to nanobyte

### Using FLA library
The flash-linear-attention (FLA) library provides optimized Triton kernels for both Mamba2 SSD and Gated DeltaNet. Rather than reimplementing the core kernels, nanobyte should use FLA as a dependency:
- `fla.ops.gated_delta_rule.chunk_gated_delta_rule` — chunked training kernel
- `fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule` — fused recurrent for inference
- Also has naive PyTorch reference implementations for education

### Key implementation notes:
1. **State shape:** (B, H, K, V) — note K and V can differ (unlike Mamba2's (B, H, P, N))
2. **Q/K are L2-normalized** — critical for training stability
3. **β (delta strength)** is a separate learned parameter via sigmoid projection
4. **g (gating)** uses Mamba2-style parameterization: g = -exp(A_log) * softplus(a_proj(x) + dt_bias)
5. **Separate convolutions** on Q, K, V (not grouped like Mamba2)

### For byte-level (MambaByte-style):
- The delta rule's error-correcting memory should help with byte-level retrieval — bytes need precise pattern matching over long sequences
- GDN's gating prevents memory saturation over 8K+ byte sequences
- Hybrid with sliding-window attention (GDN-H1) particularly interesting for byte-level where local patterns matter

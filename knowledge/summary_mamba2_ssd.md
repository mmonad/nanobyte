# Mamba-2: Transformers are SSMs (State Space Duality)

**Paper:** Tri Dao, Albert Gu (2024). "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality." arXiv:2405.21060

## Core Insight: State Space Duality (SSD)

The paper shows that SSMs and attention are **dual views of the same computation**:
- SSMs = semiseparable matrix multiplication (linear time, sequential)
- Attention = naive matrix multiplication (quadratic time, parallel/matmul-friendly)
- SSD = block decomposition that gets the best of both worlds

The key simplification from Mamba1 → Mamba2:
- **A matrix: diagonal → scalar×identity** (one scalar `a_t` per head, not per channel)
- **Head dimension: P=1 → P=64/128** (like attention heads)

This allows the quadratic "attention-like" form: `Y = (L ⊙ CB^T) · X`, where `L` is a 1-semiseparable mask from cumulative products of `a_t` values.

## SSD Algorithm (Block Decomposition)

Split sequence into chunks of length Q. For each chunk:

1. **Diagonal blocks (intra-chunk):** Compute via quadratic attention-like form within each chunk. This is a small matmul: `C^T B` gives (Q,Q) attention matrix, mask with L, multiply by X.

2. **Off-diagonal blocks (inter-chunk):** Factor into 3 terms via low-rank structure:
   - **Right (B-factors):** Compute chunk-local hidden states = BX weighted by decay → `(N, P)` per chunk
   - **Center (A-factors):** SSM scan across chunk boundaries (length T/Q, tiny cost)
   - **Left (C-factors):** Project states back to output via C

3. **Combine:** `Y = Y_diag + Y_off`

### Reference Implementation (from paper listing)
```python
def ssd(X, A, B, C, block_len=64):
    # X: (B, L, H, P), A: (B, L, H), B/C: (B, L, H, N)
    # Chunk everything
    X, A, B, C = [rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)]
    A_cumsum = cumsum(A rearranged to b h c l)

    # 1. Intra-chunk: quadratic attention
    L = exp(segsum(A))  # (Q, Q) causal mask per chunk
    Y_diag = einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

    # 2. Chunk states (B-factors)
    decay_states = exp(A_cumsum[:,:,:,-1:] - A_cumsum)
    states = einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # 3. Inter-chunk scan (A-factors) — just a small scan over T/Q steps
    decay_chunk = exp(segsum(pad(A_cumsum[:,:,:,-1])))
    new_states = einsum("bhzc,bchpn->bzhpn", decay_chunk, states)

    # 4. State → output (C-factors)
    state_decay_out = exp(A_cumsum)
    Y_off = einsum("bclhn,bchpn,bhcl->bclhp", C, states, state_decay_out)

    return Y_diag + Y_off
```

## Mamba-2 Architecture Changes (vs Mamba-1)

| Aspect | Mamba-1 | Mamba-2 |
|--------|---------|---------|
| A structure | Diagonal (D_inner, N) | Scalar × I (nheads,) |
| Head dim P | 1 (every channel independent) | 64 or 128 |
| d_state N | 16 | 64-128 (much larger!) |
| Projections | Sequential (x_proj after conv) | **Parallel** (all from input u) |
| Normalization | None before out_proj | **RMSNorm(gated)** before out_proj |
| Head pattern | MIS/MVA (B,C shared, X has heads) | Same (MVA best in ablations) |
| dt projection | Low-rank (dt_rank→d_inner) | Direct (nheads, one per head) |
| Core compute | Associative scan | **Chunked SSD (matmul-based)** |
| Speed | 1x | **2-8x faster** |

### Block Signal Flow (Mamba-2)
```
u ─── in_proj ──┬── z (gate branch, d_inner)
                ├── x (signal, d_inner)     ─┐
                ├── B (d_state per group)     ├── conv1d + SiLU ──┐
                ├── C (d_state per group)     ─┘                  │
                └── dt (nheads, + dt_bias → softplus)             │
                                                                  │
                    SSD(x, dt*A, B, C) ──── y                     │
                                             │                    │
                    RMSNorm(y, gate=z) ───── out_proj ─── output
```

Key: B and C go through conv1d together with x (not produced after conv like Mamba-1).

### Actual Dimensions (from official code)
```python
d_in_proj = 2*d_inner + 2*ngroups*d_state + nheads  # [z, x, B, C, dt]
conv_dim = d_inner + 2*ngroups*d_state               # [x, B, C] through conv
# After conv+silu, split into x (d_inner), B (ngroups*d_state), C (ngroups*d_state)
```

## Connection to nanobyte

### What changes from our mamba1.py:
1. **A_log:** `(d_inner, d_state)` → `(nheads,)` — one scalar per head
2. **dt:** `dt_rank → d_inner` bottleneck removed → direct `(nheads,)` with bias
3. **in_proj:** Sequential `in_proj → x_proj` → single parallel `in_proj` producing [z, x, B, C, dt]
4. **Conv1d:** Only on x → on [x, B, C] together
5. **Core:** selective_scan → SSD chunked algorithm (mostly matmuls!)
6. **Normalization:** Add gated RMSNorm before out_proj
7. **Heads:** Reshape x into (B, L, H, P) with P=64/128

### Performance implications:
- SSD is 2-8x faster than Mamba1's scan because it uses matmuls (tensor cores)
- Much larger state size (N=64-128 vs 16) with minimal slowdown
- The inter-chunk scan is tiny (T/Q length) so even a simple implementation is fine
- At sequence length 2K+, SSD is faster than FlashAttention-2

### For byte-level (MambaByte):
- Byte sequences are 4x longer → SSD's linear scaling matters even more
- Larger state (N=64-128) helps with information retention over long byte sequences
- The chunked matmul approach maps well to both CUDA and ROCm (no custom kernels needed!)
- Pure PyTorch implementation possible (the paper's Listing 1 is self-contained)

### Hybrid models:
- Paper finds ~10% attention layers optimal (e.g., 6 of 64 layers)
- SSM for general sequence processing, attention for retrieval/lookup
- Could be interesting for byte-level: bytes need more pattern recognition (SSM) but also exact recall (attention)

# Mamba1 vs Mamba2 Training Comparison

Tested on AMD Radeon AI PRO R9700 (32GB GDDR6, ROCm 7.2), ClimbMix data, 100 steps.

## Training Parameters

| Parameter | Value |
|---|---|
| Depth (n_layer) | 12 |
| n_embd | 768 (aspect_ratio=64) |
| Sequence length | 8192 bytes |
| Batch size | 2 |
| Total batch size | 16,384 bytes/step |
| Compute dtype | bfloat16 |
| Optimizer | MuonAdamW (matrix_lr=0.02, ssm_lr=0.001) |
| Warmup | 20 steps |
| torch.compile | enabled |

## Architecture Differences

| | Mamba1 | Mamba2 |
|---|---|---|
| A_log shape | (d_inner, d_state) = (1536, 16) | (nheads,) = (24,) |
| d_state (N) | 16 | 64 |
| dt projection | Low-rank: dt_rank(48) → d_inner(1536) | Direct: (nheads,) + bias |
| Head dimension (P) | 1 (every channel independent) | 64 |
| nheads | N/A (D_inner=1536 independent channels) | 24 |
| Projections | Sequential (in_proj → conv → x_proj → dt_proj) | Parallel (single in_proj → [z, x, B, C, dt]) |
| Conv1d input | x only (d_inner) | [x, B, C] together (d_inner + 2*d_state) |
| Pre-output norm | None | Gated RMSNorm |
| Core algorithm | Triton selective scan (parallel scan fallback) | SSD chunked matmuls (chunk_size=64) |

## Results

| Metric | Mamba1 | Mamba2 |
|---|---|---|
| **Parameters** | 45.6M | 44.4M |
| **d_inner** | 1536 | 1536 |
| **Bytes/s** | 12,090 | 12,094 |
| **MFU** | 1.71% | 1.67% |
| **Peak VRAM** | **29.0 GB** | **22.3 GB** (-23%) |
| **Val BPB @ step 0** | 8.00 | 8.00 |
| **Val BPB @ step 50** | 2.54 | 2.44 |
| **Val BPB @ step 100** | 2.18 | 2.14 |
| **Training time (100 steps)** | 127.4s | 127.4s |

## Key Takeaways

1. **Memory**: Mamba2 uses 23% less VRAM (22.3 vs 29.0 GB). Mamba1's parallel scan materializes (B, L, D, N) tensors; SSD's chunked approach only needs (B, chunk_size, H, P) at a time.

2. **Speed**: Similar throughput at this scale (~12K bytes/s). SSD's matmuls (64×64) aren't large enough to show tensor core advantage over Mamba1's Triton scan yet — advantage grows with model size.

3. **Quality**: Comparable (2.14 vs 2.18 BPB), within noise for 100 steps. Mamba2 has 4x larger state (N=64 vs 16) which should help with longer-range dependencies.

4. **Practical impact**: The 6.7 GB VRAM savings means Mamba2 could potentially fit batch_size=3 on the R9700's 32GB where Mamba1 is already at 29 GB.

# MambaByte: Token-free Selective State Space Model

**Paper:** Wang, Gangavarapu, Yan, Rush (Cornell). arXiv:2401.13660 (COLM 2024).

## Key Idea

Replace BPE tokenization with raw byte-level modeling using Mamba SSM.
Vocab size drops from 32K to 256, sequences become ~4x longer, but Mamba's
O(1) per-step inference and linear-time training handle it.

## Architecture

| Config | MambaByte-353M | MambaByte-972M | MambaByte-1.6B |
|--------|---------------|---------------|---------------|
| Layers | 53 | 48 | 48 |
| d_model | 1,024 | 1,792 | 2,304 |
| expand | 2 | 2 | 2 |
| d_state | 16 | 16 | 16 |
| d_conv | 4 | 4 | 4 |
| dt_rank | 64 | 112 | 144 |
| Context | 8,192 bytes | 8,192 bytes | 8,192 bytes |

**Critical finding: ~2x more layers than Transformers.** A Mamba layer costs ~6d^2
FLOPs/byte (expand=2), vs ~12d^2 for a Transformer layer (expand=4 MLP + attention).
To FLOP-match a 350M Transformer (24 layers, d=1024, expand=4), MambaByte uses
53 layers with d=1024, expand=2. This is the right way to compare.

## Results

- **MambaByte-353M** beats MegaByte-758M+262M on all 5 datasets (PG19, Stories,
  Books, ArXiv, Code) while using **0.63x less compute**.
- **MambaByte-972M** achieves word-level PPL of 33.0 on PG19, competitive with
  subword models like Routing-490M (33.2) and Transformer-XL (36.3).
- Length extrapolation to 4-64x training length without degradation.
- 2.6x faster generation than MegaByte (recurrent, no KV cache).
- Far more robust to noise (typos, case changes, antspeak) than tokenized models.

## Training Details

- AdamW, beta=(0.9, 0.95), linear warmup 500 steps + cosine decay
- Batch size 48, context 8192 bytes, BF16 mixed precision
- LR 0.0004, gradient clip 0.1
- 353M trained for 80K steps (~30B bytes)
- 972M trained for 380K steps (~150B bytes)

## Relevance to nanobyte

1. **Layer count:** Our Mamba1Config should default to ~48 layers, not 12.
   The training script's --depth flag maps to n_layer, so `--depth=48` is the
   MambaByte-equivalent of nanochat's `--depth=12` GPT.

2. **Aspect ratio:** MambaByte uses expand=2 (not 4 like GPT's MLP). So for
   the same d_model, Mamba has half the parameters per layer but needs 2x layers.
   Our aspect_ratio=64 convention (n_embd = depth * 64) gives:
   - depth=48: n_embd=3072, ~1B params (too big for R9700)
   - depth=24: n_embd=1536, ~250M params (good for R9700 32GB)
   The aspect ratio may need adjusting for Mamba.

3. **Sequence length:** 8192 bytes is standard. Our default matches.

4. **Speculative decoding:** MambaByte uses a small subword Mamba as drafter
   for speculative decoding, achieving 2.6x speedup. This is a unique capability
   of byte-level models — the drafter generates subword tokens which are
   verified at the byte level.

5. **Compute matching:** When comparing MambaByte vs nanochat GPT, match on
   FLOPs-per-byte, not parameter count or layer count.

"""
Unified attention interface with automatic FA3/FlexAttention/SDPA switching.

Exports `flash_attn` with the FA3 API, but routes to the best available backend:
- Flash Attention 3 on Hopper bf16
- PyTorch FlexAttention on ROCm
- PyTorch SDPA everywhere else
"""

from types import SimpleNamespace

import torch
import torch.nn.functional as F


# =============================================================================
# Detection
# =============================================================================
def _load_flash_attention_3():
    """Try to load Flash Attention 3 (requires Hopper GPU, sm90)."""
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        # FA3 kernels are compiled for Hopper (sm90) only.
        if major != 9:
            return None
        import os
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel
        return get_kernel("varunneal/flash-attention-3").flash_attn_interface
    except Exception:
        return None


def _load_flex_attention():
    """Try to load PyTorch FlexAttention."""
    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
        return SimpleNamespace(create_block_mask=create_block_mask, flex_attention=flex_attention)
    except Exception:
        return None


def _is_rocm_runtime():
    return torch.cuda.is_available() and getattr(torch.version, "hip", None) is not None


_fa3 = _load_flash_attention_3()
_flex = _load_flex_attention()
HAS_FA3 = _fa3 is not None
HAS_FLEX_ATTENTION = _flex is not None
IS_ROCM = _is_rocm_runtime()

# Override for testing: set to "fa3", "flex", "sdpa", or None (auto)
_override_impl = None
_compiled_flex_attention = None
_flex_block_mask_cache = {}


def _resolve_backend():
    """Decide once which backend to use, based on availability, override, and dtype."""
    if _override_impl == "fa3":
        assert HAS_FA3, "Cannot override to FA3: not available on this hardware"
        return "fa3"
    if _override_impl == "flex":
        assert HAS_FLEX_ATTENTION, "Cannot override to FlexAttention: not available in this PyTorch build"
        return "flex"
    if _override_impl == "sdpa":
        return "sdpa"

    if HAS_FA3:
        # FA3 Hopper kernels only support bf16 and fp8; fp16/fp32 must use fallback.
        from nanochat.common import COMPUTE_DTYPE
        if COMPUTE_DTYPE == torch.bfloat16:
            return "fa3"

    # FlexAttention: better than SDPA for sliding windows (compiled Triton kernels)
    # Works on ROCm and non-Hopper CUDA (Blackwell sm_121, Ada sm_89, etc.)
    if HAS_FLEX_ATTENTION and torch.cuda.is_available():
        return "flex"

    return "sdpa"


def _refresh_backend_state():
    global ATTENTION_BACKEND, USE_FA3, USE_FLEX
    ATTENTION_BACKEND = _resolve_backend()
    USE_FA3 = ATTENTION_BACKEND == "fa3"
    USE_FLEX = ATTENTION_BACKEND == "flex"



_refresh_backend_state()


# =============================================================================
# FlexAttention helpers
# =============================================================================
def _normalize_device(device):
    if not isinstance(device, torch.device):
        device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _flex_mask_mod(window, q_len, kv_len):
    offset = kv_len - q_len

    def mask_mod(batch, head, q_idx, kv_idx):
        row_idx = offset + q_idx
        allowed = kv_idx <= row_idx
        if window >= 0 and window < kv_len:
            allowed = allowed & ((row_idx - kv_idx) <= window)
        return allowed

    return mask_mod


def _get_flex_block_mask(device_type, device_index, q_len, kv_len, window):
    device = torch.device(device_type, device_index)
    key = (device.type, device.index, q_len, kv_len, window)

    # Avoid tracing cache mutation into outer torch.compile(model) graphs.
    if hasattr(torch, "compiler") and torch.compiler.is_compiling():
        return _flex.create_block_mask(
            _flex_mask_mod(window, q_len, kv_len),
            B=None,
            H=None,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )

    block_mask = _flex_block_mask_cache.get(key)
    if block_mask is None:
        block_mask = _flex.create_block_mask(
            _flex_mask_mod(window, q_len, kv_len),
            B=None,
            H=None,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )
        _flex_block_mask_cache[key] = block_mask
    return block_mask


def _call_flex_attention(q, k, v, block_mask, enable_gqa):
    global _compiled_flex_attention

    # If the outer model is already under torch.compile, call flex_attention directly
    # so it lowers as part of that graph. Otherwise keep an independently compiled path
    # for uncompiled eval/generation.
    if hasattr(torch, "compiler") and torch.compiler.is_compiling():
        return _flex.flex_attention(q, k, v, block_mask=block_mask, enable_gqa=enable_gqa)

    if _compiled_flex_attention is None:
        _compiled_flex_attention = torch.compile(_flex.flex_attention, dynamic=False)

    return _compiled_flex_attention(q, k, v, block_mask=block_mask, enable_gqa=enable_gqa)


def _flex_attention(q, k, v, window_size, enable_gqa):
    q_len = q.size(2)
    kv_len = k.size(2)
    window = window_size[0]
    device = _normalize_device(q.device)
    block_mask = _get_flex_block_mask(device.type, device.index, q_len, kv_len, window)
    return _call_flex_attention(q, k, v, block_mask=block_mask, enable_gqa=enable_gqa)


# =============================================================================
# SDPA helpers
# =============================================================================
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with sliding window support.
    q, k, v are (B, H, T, D) format.
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Full context, same length
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # Single token generation
    if Tq == 1:
        if window >= 0 and window < Tk:
            # window is "left" tokens we need to include (window + 1) keys total
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Need explicit mask for sliding window/chunk inference
    device = q.device
    # For chunk inference (Tq != Tk), is_causal is not aligned to cache position => build an explicit bool mask
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    # sliding window (left)
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)


# =============================================================================
# Public API: Same interface as FA3
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    if ATTENTION_BACKEND == "fa3":
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    # FlexAttention / SDPA backends use (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    if ATTENTION_BACKEND == "flex":
        y = _flex_attention(q, k, v, window_size, enable_gqa)
    else:
        y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)  # back to (B, T, H, D)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    Flash Attention with KV cache for inference.

    FA3 updates k_cache/v_cache in-place. Our fallback backends do the same.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    if ATTENTION_BACKEND == "fa3":
        return _fa3.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )

    # Fallback backends: manually manage KV cache
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].item()  # assume uniform position across batch

    # Insert new k, v into cache (in-place, matching FA3 behavior)
    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    # Get full cache up to current position + new tokens
    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    # Transpose to backend layout: (B, T, H, D) -> (B, H, T, D)
    q_backend = q.transpose(1, 2)
    k_backend = k_full.transpose(1, 2)
    v_backend = v_full.transpose(1, 2)

    enable_gqa = q_backend.size(1) != k_backend.size(1)
    if ATTENTION_BACKEND == "flex":
        y_backend = _flex_attention(q_backend, k_backend, v_backend, window_size, enable_gqa)
    else:
        y_backend = _sdpa_attention(q_backend, k_backend, v_backend, window_size, enable_gqa)

    return y_backend.transpose(1, 2)  # back to (B, T_new, H, D)


# =============================================================================
# Export: flash_attn module interface (drop-in replacement for FA3)
# =============================================================================
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)

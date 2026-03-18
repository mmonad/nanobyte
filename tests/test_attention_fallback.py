"""
Test Flash Attention unified interface - verify FA3 and SDPA produce identical results.

Run: python -m pytest tests/test_attention_fallback.py -v -s

Note on test structure:
    Tests are split into two classes due to dtype/device constraints:

    1. TestFA3VsSDPA: Comparison tests that run both FA3 and SDPA on the same inputs
       and verify they produce identical results. These require a Hopper GPU (FA3 only
       works on sm90+) and use bfloat16 (FA3 doesn't support float32).

    2. TestSDPAOnly: Tests that only exercise the SDPA fallback path. These can run
       on any device (CUDA, CPU, MPS) with the appropriate dtype for that device.
"""
import torch
import pytest
import nanochat.flash_attention as fa_module
from nanochat.common import COMPUTE_DTYPE
from nanochat.flash_attention import flash_attn, HAS_FA3, HAS_FLEX_ATTENTION, IS_ROCM
from nanochat.engine import KVCache


def set_impl(impl):
    """Set the implementation override ('fa3', 'flex', 'sdpa', or None for auto)."""
    fa_module._override_impl = impl
    fa_module._refresh_backend_state()


def run_impl_pair(fn, first_impl, second_impl):
    """Run a function with two implementations and return both outputs."""
    try:
        set_impl(first_impl)
        first_out = fn()
        set_impl(second_impl)
        second_out = fn()
    finally:
        set_impl(None)
    return first_out, second_out


def assert_close(t1, t2, name, atol=1e-2, rtol=1e-2):
    """Assert two tensors are close, with helpful error message."""
    max_diff = (t1 - t2).abs().max().item()
    mean_diff = (t1 - t2).abs().mean().item()
    assert torch.allclose(t1, t2, atol=atol, rtol=rtol), \
        f"{name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
    return max_diff, mean_diff


# =============================================================================
# FA3 vs SDPA comparison tests (require Hopper GPU)
# =============================================================================
@pytest.mark.skipif(not HAS_FA3, reason="FA3 required to compare implementations")
class TestFA3VsSDPA:
    """Compare FA3 and SDPA produce identical results. Requires Hopper GPU."""

    DEVICE = "cuda"
    DTYPE = torch.bfloat16

    def test_basic_causal(self):
        """Basic causal attention."""
        B, T, H, D = 2, 64, 4, 32
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))

        y_fa3, y_sdpa = run_impl_pair(run, 'fa3', 'sdpa')
        max_diff, mean_diff = assert_close(y_fa3, y_sdpa, "basic_causal")
        print(f"basic_causal: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_full_context(self):
        """Full context (window_size=-1)."""
        B, T, H, D = 2, 128, 4, 32
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(-1, -1))

        y_fa3, y_sdpa = run_impl_pair(run, 'fa3', 'sdpa')
        max_diff, mean_diff = assert_close(y_fa3, y_sdpa, "full_context")
        print(f"full_context: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_sliding_window(self):
        """Sliding window attention."""
        B, T, H, D = 2, 128, 4, 32
        window = 32
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(window, 0))

        y_fa3, y_sdpa = run_impl_pair(run, 'fa3', 'sdpa')
        max_diff, mean_diff = assert_close(y_fa3, y_sdpa, "sliding_window")
        print(f"sliding_window: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_gqa(self):
        """Group Query Attention (fewer KV heads than Q heads)."""
        B, T, D = 2, 64, 32
        n_heads = 8
        n_kv_heads = 2

        q = torch.randn(B, T, n_heads, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, n_kv_heads, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, n_kv_heads, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))

        y_fa3, y_sdpa = run_impl_pair(run, 'fa3', 'sdpa')
        max_diff, mean_diff = assert_close(y_fa3, y_sdpa, "gqa")
        print(f"gqa: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_larger_model(self):
        """Larger dimensions closer to real model."""
        B, T, H, D = 4, 256, 12, 64
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(-1, -1))

        y_fa3, y_sdpa = run_impl_pair(run, 'fa3', 'sdpa')
        max_diff, mean_diff = assert_close(y_fa3, y_sdpa, "larger_model")
        print(f"larger_model: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_kvcache_prefill(self):
        """Test prefill (inserting multiple tokens into empty cache)."""
        B, T_max, H, D = 2, 64, 4, 32
        T_prefill = 16

        q = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            k_cache = torch.zeros(B, T_max, H, D, device=self.DEVICE, dtype=self.DTYPE)
            v_cache = torch.zeros(B, T_max, H, D, device=self.DEVICE, dtype=self.DTYPE)
            cache_seqlens = torch.zeros(B, dtype=torch.int32, device=self.DEVICE)
            return flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache, k=k, v=v,
                cache_seqlens=cache_seqlens,
                causal=True, window_size=(T_max, 0)
            )

        y_fa3, y_sdpa = run_impl_pair(run, 'fa3', 'sdpa')
        max_diff, mean_diff = assert_close(y_fa3, y_sdpa, "prefill")
        print(f"prefill: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_kvcache_single_token(self):
        """Test single token generation (cache already has content)."""
        B, T_max, H, D = 2, 64, 4, 32
        T_prefill = 16

        k_init = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_init = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        q_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            k_cache = torch.zeros(B, T_max, H, D, device=self.DEVICE, dtype=self.DTYPE)
            v_cache = torch.zeros(B, T_max, H, D, device=self.DEVICE, dtype=self.DTYPE)
            k_cache[:, :T_prefill, :, :] = k_init
            v_cache[:, :T_prefill, :, :] = v_init
            cache_seqlens = torch.full((B,), T_prefill, dtype=torch.int32, device=self.DEVICE)
            return flash_attn.flash_attn_with_kvcache(
                q_single, k_cache, v_cache, k=k_single, v=v_single,
                cache_seqlens=cache_seqlens,
                causal=True, window_size=(T_max, 0)
            )

        y_fa3, y_sdpa = run_impl_pair(run, 'fa3', 'sdpa')
        max_diff, mean_diff = assert_close(y_fa3, y_sdpa, "single_token")
        print(f"single_token: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_kvcache_single_token_sliding_window(self):
        """Test single token decode with sliding window smaller than cache size.

        This catches the bug where SDPA ignores window_size during Tq=1 decode.
        When window < Tk, FA3 only attends to the last (window+1) tokens,
        but SDPA was attending to all cached tokens.
        """
        B, T_max, H, D = 2, 64, 4, 32
        T_prefill = 32  # Enough tokens to exceed window
        window = 8      # Window SMALLER than cache size

        k_init = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_init = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        q_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            k_cache = torch.zeros(B, T_max, H, D, device=self.DEVICE, dtype=self.DTYPE)
            v_cache = torch.zeros(B, T_max, H, D, device=self.DEVICE, dtype=self.DTYPE)
            k_cache[:, :T_prefill, :, :] = k_init
            v_cache[:, :T_prefill, :, :] = v_init
            cache_seqlens = torch.full((B,), T_prefill, dtype=torch.int32, device=self.DEVICE)
            return flash_attn.flash_attn_with_kvcache(
                q_single, k_cache, v_cache, k=k_single, v=v_single,
                cache_seqlens=cache_seqlens,
                causal=True, window_size=(window, 0)  # window=8 < Tk=33
            )

        y_fa3, y_sdpa = run_impl_pair(run, 'fa3', 'sdpa')
        max_diff, mean_diff = assert_close(y_fa3, y_sdpa, "single_token_sliding_window")
        print(f"single_token_sliding_window: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_backward_gradients_match(self):
        """Verify gradients are similar between FA3 and SDPA."""
        B, T, H, D = 2, 32, 4, 16

        q_data = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k_data = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_data = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            q = q_data.clone().requires_grad_(True)
            k = k_data.clone().requires_grad_(True)
            v = v_data.clone().requires_grad_(True)
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))
            loss = y.sum()
            loss.backward()
            return y.detach(), q.grad.detach(), k.grad.detach(), v.grad.detach()

        set_impl('fa3')
        y_fa3, q_grad_fa3, k_grad_fa3, v_grad_fa3 = run()
        set_impl('sdpa')
        y_sdpa, q_grad_sdpa, k_grad_sdpa, v_grad_sdpa = run()
        set_impl(None)

        max_diff, mean_diff = assert_close(y_fa3, y_sdpa, "backward_output")
        print(f"backward_output: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        max_diff, mean_diff = assert_close(q_grad_fa3, q_grad_sdpa, "q_grad", atol=0.05, rtol=0.05)
        print(f"q_grad: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        max_diff, mean_diff = assert_close(k_grad_fa3, k_grad_sdpa, "k_grad", atol=0.05, rtol=0.05)
        print(f"k_grad: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        max_diff, mean_diff = assert_close(v_grad_fa3, v_grad_sdpa, "v_grad", atol=0.05, rtol=0.05)
        print(f"v_grad: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")


# =============================================================================
# FlexAttention vs SDPA comparison tests (require ROCm)
# =============================================================================
@pytest.mark.skipif(not (IS_ROCM and HAS_FLEX_ATTENTION), reason="ROCm FlexAttention required")
class TestFlexVsSDPA:
    """Compare FlexAttention and SDPA produce similar results on ROCm."""

    DEVICE = "cuda"
    DTYPE = torch.bfloat16

    def test_basic_causal(self):
        """Basic causal attention."""
        B, T, H, D = 2, 64, 4, 32
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))

        y_flex, y_sdpa = run_impl_pair(run, 'flex', 'sdpa')
        max_diff, mean_diff = assert_close(y_flex, y_sdpa, "flex_basic_causal")
        print(f"flex_basic_causal: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_sliding_window_gqa(self):
        """Sliding window attention with GQA."""
        B, T, D = 2, 64, 32
        H, H_kv = 8, 2
        window = 16
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H_kv, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H_kv, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(window, 0))

        y_flex, y_sdpa = run_impl_pair(run, 'flex', 'sdpa')
        max_diff, mean_diff = assert_close(y_flex, y_sdpa, "flex_sliding_window_gqa")
        print(f"flex_sliding_window_gqa: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_kvcache_chunk_decode(self):
        """Chunk decode should match SDPA when cache already has content."""
        B, T_max, D = 2, 64, 32
        H, H_kv = 8, 2
        T_prefill = 16
        T_new = 3
        window = 8

        k_init = torch.randn(B, T_prefill, H_kv, D, device=self.DEVICE, dtype=self.DTYPE)
        v_init = torch.randn(B, T_prefill, H_kv, D, device=self.DEVICE, dtype=self.DTYPE)
        q_new = torch.randn(B, T_new, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k_new = torch.randn(B, T_new, H_kv, D, device=self.DEVICE, dtype=self.DTYPE)
        v_new = torch.randn(B, T_new, H_kv, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            k_cache = torch.zeros(B, T_max, H_kv, D, device=self.DEVICE, dtype=self.DTYPE)
            v_cache = torch.zeros(B, T_max, H_kv, D, device=self.DEVICE, dtype=self.DTYPE)
            k_cache[:, :T_prefill, :, :] = k_init
            v_cache[:, :T_prefill, :, :] = v_init
            cache_seqlens = torch.full((B,), T_prefill, dtype=torch.int32, device=self.DEVICE)
            return flash_attn.flash_attn_with_kvcache(
                q_new, k_cache, v_cache, k=k_new, v=v_new,
                cache_seqlens=cache_seqlens,
                causal=True, window_size=(window, 0)
            )

        y_flex, y_sdpa = run_impl_pair(run, 'flex', 'sdpa')
        max_diff, mean_diff = assert_close(y_flex, y_sdpa, "flex_kvcache_chunk_decode")
        print(f"flex_kvcache_chunk_decode: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")


# =============================================================================
# SDPA-only tests (run on any device)
# =============================================================================
class TestSDPAOnly:
    """Test SDPA fallback works correctly. Runs on any device."""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    def test_basic_forward(self):
        """Test SDPA forward pass produces valid output."""
        set_impl('sdpa')
        B, T, H, D = 2, 64, 4, 32
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))

        assert y.shape == (B, T, H, D)
        assert not torch.isnan(y).any(), "Output contains NaN"
        set_impl(None)

    def test_backward(self):
        """Test gradients flow through SDPA."""
        set_impl('sdpa')
        B, T, H, D = 2, 32, 4, 16
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE, requires_grad=True)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE, requires_grad=True)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE, requires_grad=True)

        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))
        loss = y.sum()
        loss.backward()

        assert q.grad is not None, "No gradient for q"
        assert k.grad is not None, "No gradient for k"
        assert v.grad is not None, "No gradient for v"
        assert not torch.isnan(q.grad).any(), "NaN in q gradient"
        set_impl(None)

    def test_kvcache(self):
        """Test SDPA with KV cache."""
        set_impl('sdpa')
        B, T_max, H, D = 2, 64, 4, 32
        n_layers = 1

        cache = KVCache(
            batch_size=B, num_heads=H, seq_len=T_max, head_dim=D,
            num_layers=n_layers, device=self.DEVICE, dtype=self.DTYPE
        )
        k_cache, v_cache = cache.get_layer_cache(0)

        # Prefill
        T_prefill = 16
        q = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)

        y = flash_attn.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v,
            cache_seqlens=cache.cache_seqlens,
            causal=True, window_size=(T_max, 0)
        )
        cache.advance(T_prefill)

        assert y.shape == (B, T_prefill, H, D)
        assert cache.get_pos() == T_prefill

        # Generate single token
        q_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)

        y_single = flash_attn.flash_attn_with_kvcache(
            q_single, k_cache, v_cache, k=k_single, v=v_single,
            cache_seqlens=cache.cache_seqlens,
            causal=True, window_size=(T_max, 0)
        )
        cache.advance(1)

        assert y_single.shape == (B, 1, H, D)
        assert cache.get_pos() == T_prefill + 1
        set_impl(None)


# =============================================================================
# Override mechanism tests
# =============================================================================
class TestOverrideMechanism:
    """Test that the override mechanism works correctly."""

    @pytest.mark.skipif(not HAS_FA3, reason="FA3 required")
    def test_override_fa3(self):
        """Test that override='fa3' uses FA3."""
        set_impl('fa3')
        assert fa_module.ATTENTION_BACKEND == 'fa3'
        assert fa_module.USE_FA3 == True
        set_impl(None)

    @pytest.mark.skipif(not (IS_ROCM and HAS_FLEX_ATTENTION), reason="ROCm FlexAttention required")
    def test_override_flex(self):
        """Test that override='flex' uses FlexAttention."""
        set_impl('flex')
        assert fa_module.ATTENTION_BACKEND == 'flex'
        assert fa_module.USE_FLEX == True
        assert fa_module.USE_FA3 == False
        set_impl(None)

    def test_override_sdpa(self):
        """Test that override='sdpa' uses SDPA."""
        set_impl('sdpa')
        assert fa_module.ATTENTION_BACKEND == 'sdpa'
        assert fa_module.USE_FA3 == False
        set_impl(None)

    def test_override_auto(self):
        """Test that override=None uses auto-detection."""
        set_impl(None)
        if HAS_FA3 and COMPUTE_DTYPE == torch.bfloat16:
            expected = 'fa3'
        elif IS_ROCM and HAS_FLEX_ATTENTION:
            expected = 'flex'
        else:
            expected = 'sdpa'
        assert fa_module.ATTENTION_BACKEND == expected
        assert fa_module.USE_FA3 == (expected == 'fa3')
        assert fa_module.USE_FLEX == (expected == 'flex')


if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        major, minor = torch.cuda.get_device_capability()
        print(f"Compute capability: {major}.{minor}")
    print(f"HAS_FA3: {HAS_FA3}")
    print(f"HAS_FLEX_ATTENTION: {HAS_FLEX_ATTENTION}")
    print(f"IS_ROCM: {IS_ROCM}")
    print(f"ATTENTION_BACKEND: {fa_module.ATTENTION_BACKEND}")
    print()

    pytest.main([__file__, "-v", "-s"])

"""Basic tests for Mamba2 SSD implementation."""

import torch
import pytest
from nanochat.mamba2 import Mamba2, Mamba2Config, ssd_forward, segsum, norm


def test_segsum_shape():
    """segsum should produce (*, T, T) lower-triangular decay matrix."""
    x = torch.randn(2, 4, 3, 8)  # (B, H, nchunks, chunk_size)
    result = segsum(x)
    assert result.shape == (2, 4, 3, 8, 8)
    # Upper triangle should be -inf
    mask = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)
    assert (result[..., mask] == -torch.inf).all()


def test_segsum_diagonal():
    """Diagonal of segsum should be 0 (no decay from position to itself)."""
    x = torch.randn(1, 1, 1, 4)
    result = segsum(x)
    diag = torch.diagonal(result[0, 0, 0])
    assert torch.allclose(diag, torch.zeros_like(diag))


def test_ssd_forward_shape():
    """SSD forward should produce correct output shape."""
    B, L, H, P, N = 2, 128, 4, 32, 16
    X = torch.randn(B, L, H, P)
    A = torch.randn(B, L, H) * -0.1  # small negative
    B_ssm = torch.randn(B, L, 1, N)  # ngroups=1
    C = torch.randn(B, L, 1, N)
    Y, final_state = ssd_forward(X, A, B_ssm, C, chunk_size=32)
    assert Y.shape == (B, L, H, P)
    assert final_state.shape == (B, 1, H, P, N)


def test_ssd_causal():
    """Output at position t should not depend on inputs after t."""
    B, L, H, P, N = 1, 64, 2, 16, 8
    X = torch.randn(B, L, H, P, requires_grad=True)
    A = torch.randn(B, L, H) * -0.1
    B_ssm = torch.randn(B, L, 1, N)
    C = torch.randn(B, L, 1, N)

    Y, _ = ssd_forward(X, A, B_ssm, C, chunk_size=16)
    # Check: gradient of Y[0, 0, :, :] w.r.t. X should be zero for positions > 0
    Y[0, 0].sum().backward()
    # X.grad for positions 1+ should be zero (causal)
    assert X.grad[:, 1:].abs().max() < 1e-5


def test_model_forward():
    """Full model forward pass should produce a scalar loss."""
    config = Mamba2Config(
        sequence_len=128, vocab_size=256, n_layer=2,
        n_embd=64, expand=2, d_state=16, headdim=32, chunk_size=32,
    )
    model = Mamba2(config)
    model.init_weights()
    idx = torch.randint(0, 256, (2, 128))
    targets = torch.randint(0, 256, (2, 128))
    loss = model(idx, targets)
    assert loss.shape == ()
    assert loss.item() > 0


def test_model_forward_padding():
    """Model should handle sequence lengths not divisible by chunk_size."""
    config = Mamba2Config(
        sequence_len=100, vocab_size=256, n_layer=2,
        n_embd=64, expand=2, d_state=16, headdim=32, chunk_size=32,
    )
    model = Mamba2(config)
    model.init_weights()
    # 100 is not divisible by 32 — should pad internally
    idx = torch.randint(0, 256, (1, 100))
    targets = torch.randint(0, 256, (1, 100))
    loss = model(idx, targets)
    assert loss.shape == ()


def test_model_generate():
    """Generation should yield bytes."""
    config = Mamba2Config(
        sequence_len=64, vocab_size=256, n_layer=2,
        n_embd=64, expand=2, d_state=16, headdim=32, chunk_size=16,
    )
    model = Mamba2(config)
    model.init_weights()
    tokens = list(b"Hello")
    generated = list(model.generate(tokens, max_tokens=5))
    assert len(generated) == 5
    assert all(0 <= t < 256 for t in generated)


def test_ssd_forward_ngroups():
    """SSD forward should work with ngroups > 1."""
    B, L, H, P, N, G = 2, 64, 4, 16, 8, 2
    X = torch.randn(B, L, H, P)
    A = torch.randn(B, L, H) * -0.1
    B_ssm = torch.randn(B, L, G, N)
    C = torch.randn(B, L, G, N)
    Y, final_state = ssd_forward(X, A, B_ssm, C, chunk_size=16)
    assert Y.shape == (B, L, H, P)
    assert final_state.shape == (B, 1, H, P, N)


def test_ssd_initial_states_roundtrip():
    """final_state from one call should be usable as initial_states for the next."""
    B, L, H, P, N = 1, 64, 2, 16, 8
    X1 = torch.randn(B, L, H, P)
    X2 = torch.randn(B, L, H, P)
    A1 = torch.randn(B, L, H) * -0.1
    A2 = torch.randn(B, L, H) * -0.1
    B_ssm = torch.randn(B, L, 1, N)
    C = torch.randn(B, L, 1, N)

    _, final1 = ssd_forward(X1, A1, B_ssm, C, chunk_size=16)
    Y2, _ = ssd_forward(X2, A2, B_ssm, C, chunk_size=16, initial_states=final1)
    assert Y2.shape == (B, L, H, P)


def test_forward_step_parity():
    """Mixer forward and step should produce the same output."""
    config = Mamba2Config(
        sequence_len=64, vocab_size=256, n_layer=1,
        n_embd=64, expand=2, d_state=16, headdim=32, chunk_size=16,
    )
    model = Mamba2(config)
    model.init_weights()
    model.eval()

    idx = torch.randint(0, 256, (1, 64))
    with torch.no_grad():
        logits_fwd = model(idx)

    cfg = config
    conv_dim = cfg.d_inner + 2 * cfg.ngroups * cfg.d_state
    conv_states = [torch.zeros(1, conv_dim, cfg.d_conv) for _ in range(cfg.n_layer)]
    ssm_states = [torch.zeros(1, cfg.nheads, cfg.headdim, cfg.d_state) for _ in range(cfg.n_layer)]

    with torch.no_grad():
        for t in range(64):
            x = model.wte(idx[:, t:t+1]).squeeze(1)
            x = norm(x)
            for i, layer in enumerate(model.layers):
                x = layer.step(x, conv_states[i], ssm_states[i])
            x = norm(x)
        logits_step = model.lm_head(x)[:, :cfg.vocab_size].float()
        softcap = 15
        logits_step = softcap * torch.tanh(logits_step / softcap)

    logits_fwd_last = logits_fwd[:, -1, :]
    atol = 0.05
    assert torch.allclose(logits_fwd_last, logits_step, atol=atol), \
        f"Max diff: {(logits_fwd_last - logits_step).abs().max().item()}"


def test_param_count():
    """num_scaling_params should account for all parameters."""
    config = Mamba2Config(
        sequence_len=64, vocab_size=256, n_layer=4,
        n_embd=128, expand=2, d_state=32, headdim=64, chunk_size=16,
    )
    model = Mamba2(config)
    model.init_weights()
    counts = model.num_scaling_params()
    assert counts['total'] == sum(p.numel() for p in model.parameters())

"""
Mamba2 byte-level language model (MambaByte style, SSD algorithm).
Notable features:
- byte-level: vocab_size=256, no tokenizer needed
- State Space Duality (SSD): chunked computation using matmuls, 2-8x faster than Mamba1
- scalar A per head (not diagonal per channel), enabling attention-like quadratic form
- multi-head structure with head dimension P (like attention heads)
- parallel projections: all of (z, x, B, C, dt) produced from input in one shot
- gated RMSNorm before output projection (improves stability at scale)
- O(1) inference memory per layer (no KV cache growth)
- depth is the only dial (same philosophy as nanochat GPT)
"""

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW

try:
    from fla.ops.simple_gla import chunk_simple_gla
    # FLA Triton kernels fail to compile on RDNA4 (gfx1201) due to ROCm Triton backend limitations.
    # Forward pass works but backward pass compilation crashes. Use pure PyTorch fallback on ROCm.
    HAS_FLA = not (torch.cuda.is_available() and getattr(torch.version, "hip", None) is not None)
except ImportError:
    HAS_FLA = False


# ─── Config ─────────────────────────────────────────────────────────────────

@dataclass
class Mamba2Config:
    sequence_len: int = 8192  # byte-level needs longer sequences (~4x token-level)
    vocab_size: int = 256     # raw UTF-8 bytes
    n_layer: int = 24         # ~2x GPT layers (Mamba layers are ~half the FLOPs each)
    n_embd: int = 768         # d_model
    expand: int = 2           # d_inner = expand * n_embd
    d_state: int = 64         # SSM state dimension N (much larger than Mamba1's 16)
    d_conv: int = 4           # causal conv1d kernel width
    headdim: int = 64         # head dimension P (like attention heads)
    ngroups: int = 1          # number of B,C head groups (1 = MVA pattern)
    chunk_size: int = 64      # SSD block/chunk length Q

    def __post_init__(self):
        self.d_inner = self.expand * self.n_embd
        assert self.d_inner % self.headdim == 0, "d_inner must be divisible by headdim"
        self.nheads = self.d_inner // self.headdim


# ─── Utilities ──────────────────────────────────────────────────────────────

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype),
                        self.bias.to(dtype=x.dtype) if self.bias is not None else None)


# ─── SSD Algorithm ─────────────────────────────────────────────────────────
# The State Space Dual (SSD) algorithm computes SSMs via block decomposition:
#   - Intra-chunk: quadratic attention-like matmuls (diagonal blocks)
#   - Inter-chunk: small SSM scan across chunk boundaries (off-diagonal blocks)
# This makes ~95% of compute matmul-based (tensor core friendly).

def segsum(x):
    """
    Stable segment sum for computing 1-semiseparable mask L = exp(segsum(A)).
    Given A values within a chunk, computes the cumulative decay matrix where
    L[i,j] = exp(A[j+1] + A[j+2] + ... + A[i]) for i >= j, else -inf.
    """
    T = x.size(-1)
    x = x.unsqueeze(-1).expand(*x.shape, T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd_forward(X, A, B, C, chunk_size, initial_states=None):
    """
    State Space Dual (SSD) forward pass — the core of Mamba2.

    This is a pure PyTorch implementation of the SSD algorithm (Listing 1 from the paper).
    It splits the sequence into chunks and uses:
    - Quadratic attention within chunks (matmul-friendly)
    - Linear SSM scan between chunks (tiny, T/Q steps)

    Args:
        X: (B, L, H, P) — input values (like V in attention)
        A: (B, L, H)    — log-space decay scalars (already dt*A, negative)
        B: (B, L, G, N) — input projection (like K in attention)
        C: (B, L, G, N) — output projection (like Q in attention)
        chunk_size: int  — block length Q
        initial_states: optional (B, 1, H, P, N) initial hidden states

    Returns:
        Y: (B, L, H, P)           — output
        final_state: (B, 1, H, P, N) — final hidden state (can be fed back as initial_states)
    """
    batch, seqlen, nheads, headdim = X.shape
    ngroups = B.shape[2]
    dstate = B.shape[3]
    assert seqlen % chunk_size == 0
    nchunks = seqlen // chunk_size

    # Reshape into chunks: (B, nchunks, chunk_size, ...)
    X = X.reshape(batch, nchunks, chunk_size, nheads, headdim)
    A = A.reshape(batch, nchunks, chunk_size, nheads)
    B = B.reshape(batch, nchunks, chunk_size, ngroups, dstate)
    C = C.reshape(batch, nchunks, chunk_size, ngroups, dstate)

    # Rearrange A to (B, H, nchunks, chunk_size) for cumsum
    A = A.permute(0, 3, 1, 2)  # (B, H, nchunks, chunk_size)
    A_cumsum = torch.cumsum(A, dim=-1)

    # Expand B,C groups to match heads: (B, nchunks, chunk_size, H, N)
    heads_per_group = nheads // ngroups
    if ngroups < nheads:
        B = B.unsqueeze(4).expand(*B.shape[:3], ngroups, heads_per_group, dstate)
        B = B.reshape(batch, nchunks, chunk_size, nheads, dstate)
        C = C.unsqueeze(4).expand(*C.shape[:3], ngroups, heads_per_group, dstate)
        C = C.reshape(batch, nchunks, chunk_size, nheads, dstate)
    else:
        # ngroups == nheads, already (B, nchunks, chunk_size, H, N)
        pass

    # ── Step 1: Intra-chunk output (diagonal blocks) ──
    # Compute quadratic attention within each chunk
    # L[i,j] = exp(A[i] + A[i-1] + ... + A[j+1]) for causal positions
    L = torch.exp(segsum(A))  # (B, H, nchunks, chunk_size, chunk_size)

    # Y_diag = einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)
    # C: (B,nc,Q,H,N), B: (B,nc,Q,H,N), L: (B,H,nc,Q,Q), X: (B,nc,Q,H,P)
    # Step: CB^T → (B,nc,H,Q,Q), mask with L, multiply by X
    CB = torch.einsum("bclhn,bcshn->bhcls", C, B)  # (B, H, nc, Q, Q)
    CB_masked = CB * L                               # apply causal decay mask
    Y_diag = torch.einsum("bhcls,bcshp->bclhp", CB_masked, X)

    # ── Step 2: Compute chunk states (B-factors, right term) ──
    # Final state per chunk assuming zero initial state
    # decay_states[t] = exp(A_cumsum[-1] - A_cumsum[t]) = decay from t to end of chunk
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)  # (B, H, nc, Q)
    # states = einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # ── Step 3: Inter-chunk SSM scan (A-factors, center term) ──
    # Propagate states across chunk boundaries
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)  # (B, nc+1, H, P, N)

    # Decay between chunks: cumulative product of end-of-chunk A values
    chunk_decay_vals = F.pad(A_cumsum[:, :, :, -1], (1, 0))  # (B, H, nc+1)
    decay_chunk = torch.exp(segsum(chunk_decay_vals))  # (B, H, nc+1, nc+1)
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states = new_states[:, :-1]        # (B, nc, H, P, N)
    final_state = new_states[:, -1:]   # (B, 1, H, P, N) — keep 5D for round-trip

    # ── Step 4: State → output conversion (C-factors, left term) ──
    state_decay_out = torch.exp(A_cumsum)  # (B, H, nc, Q)
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C.float(), states, state_decay_out)

    # Combine intra-chunk and inter-chunk outputs
    Y = (Y_diag + Y_off).to(X.dtype).reshape(batch, seqlen, nheads, headdim)
    return Y, final_state


def ssd_step(x, dt, A, B, C, D, ssm_state):
    """
    Single-step recurrent inference for SSD (same recurrence as any SSM).
    Much simpler than the chunked training path — just the scalar SSM update.

    Args:
        x:  (B, H, P) — input values
        dt: (B, H)    — timestep deltas (after softplus)
        A:  (H,)      — state decay (negative)
        B:  (B, H, N) — input projection (already expanded to per-head)
        C:  (B, H, N) — output projection (already expanded to per-head)
        D:  (H,)      — skip connection
        ssm_state: (B, H, P, N) — mutable, updated in place
    Returns:
        y: (B, H, P)
    """
    # Discretize: dA = exp(dt * A), shape (B, H)
    dA = torch.exp(dt * A)
    # dBx = dt * B * x, shape (B, H, P, N)
    dBx = torch.einsum("bh,bhn,bhp->bhpn", dt, B, x)
    # State update: h = dA * h + dBx
    ssm_state.copy_(ssm_state * dA.unsqueeze(-1).unsqueeze(-1) + dBx)
    # Output: y = C^T h + D * x
    y = torch.einsum("bhpn,bhn->bhp", ssm_state.to(x.dtype), C)
    y = y + D.unsqueeze(-1) * x
    return y


# ─── Model Components ──────────────────────────────────────────────────────

class Mamba2Mixer(nn.Module):
    """
    Mamba2 SSD mixer with parallel projections and gated RMSNorm.

    Signal flow (parallel projections — key difference from Mamba1):
        input ── in_proj ──┬── z (gate, d_inner)
                            ├── x (signal, d_inner)  ─┐
                            ├── B (ngroups * d_state)  ├── conv1d → SiLU
                            ├── C (ngroups * d_state)  ─┘
                            └── dt (nheads, + bias → softplus)
                                        │
                    SSD(x_heads, dt*A, B, C) → y
                                        │
                    RMSNorm(y, gate=z) → out_proj → output
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.nheads = config.nheads
        self.headdim = config.headdim
        self.d_inner = config.d_inner
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.ngroups = config.ngroups
        self.chunk_size = config.chunk_size

        # Single parallel projection: [z, x, B, C, dt] all from input
        d_in_proj = (2 * config.d_inner                     # z + x
                     + 2 * config.ngroups * config.d_state   # B + C
                     + config.nheads)                        # dt
        self.in_proj = Linear(config.n_embd, d_in_proj, bias=False)

        # Conv1d on [x, B, C] together (not just x like Mamba1)
        conv_dim = config.d_inner + 2 * config.ngroups * config.d_state
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, kernel_size=config.d_conv,
                                groups=conv_dim, padding=config.d_conv - 1, bias=True)

        # Output projection
        self.out_proj = Linear(config.d_inner, config.n_embd, bias=False)

        # SSM parameters (initialized in Mamba2.init_weights)
        self.A_log = nn.Parameter(torch.empty(config.nheads))
        self.D = nn.Parameter(torch.empty(config.nheads))
        self.dt_bias = nn.Parameter(torch.empty(config.nheads))

        # Gated RMSNorm before out_proj (key Mamba2 architectural addition)
        self.norm = nn.Parameter(torch.empty(config.d_inner))

    def _gated_rmsnorm(self, y, z):
        """RMSNorm(y) * SiLU(z) — gated normalization before output projection."""
        y_normed = F.rms_norm(y, (y.size(-1),))
        return y_normed * self.norm.to(y.dtype) * F.silu(z)

    def forward(self, u):
        """u: (B, L, D) → (B, L, D)"""
        batch, seqlen, _ = u.shape

        # ── Parallel projections (all from input u) ──
        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        z, xBC, dt = zxbcdt.split(
            [self.d_inner,
             self.d_inner + 2 * self.ngroups * self.d_state,
             self.nheads], dim=-1
        )

        # ── Conv1d on [x, B, C] together ──
        xBC = xBC.transpose(1, 2)
        xBC = F.conv1d(xBC, self.conv1d.weight.to(dtype=xBC.dtype),
                       self.conv1d.bias.to(dtype=xBC.dtype),
                       padding=self.conv1d.padding, groups=self.conv1d.groups)
        xBC = xBC[:, :, :seqlen].transpose(1, 2)
        xBC = F.silu(xBC)

        # Split into x, B, C
        x, B, C = xBC.split(
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1
        )

        # ── Compute dt and A ──
        dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)
        A = -torch.exp(self.A_log.float())  # (nheads,) — always negative

        # ── Reshape for SSD ──
        x = x.reshape(batch, seqlen, self.nheads, self.headdim)  # (B, L, H, P)
        B = B.reshape(batch, seqlen, self.ngroups, self.d_state)  # (B, L, G, N)
        C = C.reshape(batch, seqlen, self.ngroups, self.d_state)  # (B, L, G, N)

        # Discretize: input *= dt, decay = A * dt (log-space)
        x_dt = x * dt.unsqueeze(-1)                   # (B, L, H, P)
        g = A.unsqueeze(0).unsqueeze(0) * dt           # (B, L, H) — log-space decay

        # Expand B,C groups to match heads for FLA kernel
        if self.ngroups < self.nheads:
            heads_per_group = self.nheads // self.ngroups
            B = B.unsqueeze(3).expand(*B.shape[:2], self.ngroups, heads_per_group, self.d_state)
            B = B.reshape(batch, seqlen, self.nheads, self.d_state)
            C = C.unsqueeze(3).expand(*C.shape[:2], self.ngroups, heads_per_group, self.d_state)
            C = C.reshape(batch, seqlen, self.nheads, self.d_state)

        # ── SSD core ──
        # FLA mapping: SSD(X, A, B, C) → simple_gla(q=C, k=B, v=X, g=A)
        if HAS_FLA and x.is_cuda:
            y, _ = chunk_simple_gla(q=C, k=B, v=x_dt, g=g, scale=1.0)
        else:
            y, _ = ssd_forward(x_dt, g, B, C, chunk_size=self.chunk_size)

        # ── D skip connection ──
        y = y + x * self.D.unsqueeze(-1)

        # ── Gated RMSNorm + output projection ──
        y = y.reshape(batch, seqlen, self.d_inner)
        y = self._gated_rmsnorm(y, z)
        return self.out_proj(y)

    def step(self, u, conv_state, ssm_state):
        """
        Single-step recurrent inference. O(1) per step.
        u: (B, D) → (B, D)
        conv_state: (B, conv_dim, d_conv) — mutable
        ssm_state:  (B, H, P, N) — mutable
        """
        # Parallel projections
        zxbcdt = self.in_proj(u)
        z, xBC, dt = zxbcdt.split(
            [self.d_inner,
             self.d_inner + 2 * self.ngroups * self.d_state,
             self.nheads], dim=-1
        )

        # Conv step: shift state, insert new input
        conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
        conv_state[:, :, -1] = xBC
        conv_w = self.conv1d.weight.squeeze(1).to(dtype=xBC.dtype)
        xBC = (conv_state * conv_w).sum(-1)
        if self.conv1d.bias is not None:
            xBC = xBC + self.conv1d.bias.to(dtype=xBC.dtype)
        xBC = F.silu(xBC)

        # Split
        x, B, C = xBC.split(
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1
        )

        # SSM step
        dt = F.softplus(dt + self.dt_bias)  # (B, nheads)
        A = -torch.exp(self.A_log.float())  # (nheads,)
        x = x.reshape(-1, self.nheads, self.headdim)

        # Expand B,C from groups to per-head (ssd_step expects (B, H, N))
        heads_per_group = self.nheads // self.ngroups
        B = B.reshape(-1, self.ngroups, self.d_state)
        B = B.repeat_interleave(heads_per_group, dim=1)  # (batch, H, N)
        C = C.reshape(-1, self.ngroups, self.d_state)
        C = C.repeat_interleave(heads_per_group, dim=1)  # (batch, H, N)

        y = ssd_step(x, dt, A, B, C, self.D, ssm_state)

        # Gated RMSNorm + output projection
        y = y.reshape(-1, self.d_inner)
        y = self._gated_rmsnorm(y, z)
        return self.out_proj(y)


class Mamba2Block(nn.Module):
    """Pre-norm residual block: x + Mixer(norm(x))"""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.mixer = Mamba2Mixer(config, layer_idx)

    def forward(self, x):
        return x + self.mixer(norm(x))

    def step(self, x, conv_state, ssm_state):
        return x + self.mixer.step(norm(x), conv_state, ssm_state)


# ─── Full Model ─────────────────────────────────────────────────────────────

class Mamba2(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE: runs in meta device context. Actual initialization in init_weights().
        """
        super().__init__()
        self.config = config
        padded_vocab = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab}")
        self.padded_vocab = padded_vocab

        self.wte = nn.Embedding(padded_vocab, config.n_embd)
        self.layers = nn.ModuleList([Mamba2Block(config, i) for i in range(config.n_layer)])
        self.lm_head = Linear(config.n_embd, padded_vocab, bias=False)

    @torch.no_grad()
    def init_weights(self):
        """Initialize all parameters. Called after .to_empty(device)."""
        cfg = self.config
        d, d_inner = cfg.n_embd, cfg.d_inner

        # ── Embedding + LM head ──
        nn.init.normal_(self.wte.weight, std=0.8)
        nn.init.normal_(self.lm_head.weight, std=0.001)

        # ── Per-layer initialization ──
        s = 3**0.5 * d**-0.5

        for block in self.layers:
            m = block.mixer

            # in_proj: uniform matching normal std at d_model scale
            nn.init.uniform_(m.in_proj.weight, -s, s)

            # out_proj: zero init (residual starts neutral)
            nn.init.zeros_(m.out_proj.weight)

            # A_log: uniform in [1, 16], then log (per the paper)
            A = torch.empty(cfg.nheads, device=m.A_log.device).uniform_(1, 16)
            m.A_log.copy_(torch.log(A))

            # D: skip connection, initialize to 1
            m.D.fill_(1.0)

            # dt_bias: inv_softplus of uniform in [dt_min, dt_max] log-space
            dt = torch.exp(
                torch.rand(cfg.nheads, device=m.dt_bias.device)
                * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
            ).clamp(min=1e-4)
            m.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inv_softplus

            # Gated RMSNorm weight: initialize to 1
            m.norm.fill_(1.0)

            # conv1d: kaiming uniform (matches nn.Conv1d default)
            fan_in = m.conv1d.in_channels // m.conv1d.groups * m.conv1d.kernel_size[0]
            nn.init.kaiming_uniform_(m.conv1d.weight, a=math.sqrt(5))
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(m.conv1d.bias, -bound, bound)

        # Cast embedding to compute dtype (bf16 usually)
        if COMPUTE_DTYPE != torch.float16:
            self.wte.to(dtype=COMPUTE_DTYPE)

    def get_device(self):
        return self.wte.weight.device

    def estimate_flops(self):
        """
        Estimated FLOPs per token (forward + backward).
        Each matmul parameter contributes 6 FLOPs (2 fwd, 4 bwd).
        Non-matmul ops (SSD scan, conv1d, elementwise) are small and excluded.
        """
        nparams = sum(p.numel() for p in self.parameters())
        nparams_exclude = self.wte.weight.numel()
        for block in self.layers:
            m = block.mixer
            nparams_exclude += (m.A_log.numel() + m.D.numel() + m.dt_bias.numel()
                                + m.norm.numel()
                                + m.conv1d.weight.numel() + m.conv1d.bias.numel())
        return 6 * (nparams - nparams_exclude)

    def num_scaling_params(self):
        """Parameter counts by group for scaling law analysis."""
        wte = self.wte.weight.numel()
        lm_head = self.lm_head.weight.numel()
        layer_matmul = sum(
            m.mixer.in_proj.weight.numel() + m.mixer.out_proj.weight.numel()
            for m in self.layers)
        layer_ssm = sum(
            m.mixer.A_log.numel() + m.mixer.D.numel() + m.mixer.dt_bias.numel()
            for m in self.layers)
        layer_norm = sum(m.mixer.norm.numel() for m in self.layers)
        layer_conv = sum(
            m.mixer.conv1d.weight.numel() + m.mixer.conv1d.bias.numel()
            for m in self.layers)
        total = wte + lm_head + layer_matmul + layer_ssm + layer_norm + layer_conv
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'lm_head': lm_head,
            'transformer_matrices': layer_matmul,
            'layer_ssm': layer_ssm,
            'layer_norm': layer_norm,
            'layer_conv': layer_conv,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.008, embedding_lr=0.3, matrix_lr=0.02,
                        ssm_lr=0.001, conv_lr=0.01, weight_decay=0.0):
        """
        Set up MuonAdamW optimizer with Mamba2-appropriate parameter groups.

        - Muon for 2D weight matrices (in_proj, out_proj)
        - AdamW for embeddings, LM head, SSM params (A_log, D, dt_bias), conv, norm
        """
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # ── Collect parameter groups ──
        matrix_params = []
        ssm_params = []
        conv_params = []
        norm_params = []

        for block in self.layers:
            m = block.mixer
            matrix_params.extend([m.in_proj.weight, m.out_proj.weight])
            ssm_params.extend([m.A_log, m.D, m.dt_bias])
            conv_params.extend([m.conv1d.weight, m.conv1d.bias])
            norm_params.append(m.norm)

        embedding_params = list(self.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())

        # Verify completeness
        n_total = (len(matrix_params) + len(ssm_params) + len(conv_params)
                   + len(norm_params) + len(embedding_params) + len(lm_head_params))
        assert n_total == len(list(self.parameters())), \
            f"Parameter group mismatch: {n_total} grouped vs {len(list(self.parameters()))} total"

        # Scale LR for AdamW params ∝ 1/√(d_model/768)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling AdamW LRs ∝ 1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # ── AdamW groups ──
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale,
                 betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale,
                 betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=ssm_params, lr=ssm_lr,
                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0),
            dict(kind='adamw', params=conv_params, lr=conv_lr,
                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01),
            dict(kind='adamw', params=norm_params, lr=ssm_lr,
                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0),
        ]

        # ── Muon groups (matrix params, grouped by shape for stacked Newton-Schulz) ──
        for shape in sorted({p.shape for p in matrix_params}):
            group = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, loss_reduction='mean'):
        """
        Forward pass. Returns loss if targets given, else logits.
        idx: (B, T) int64 byte values [0, 255]
        targets: (B, T) int64 byte values or None
        """
        B, T = idx.shape

        # Pad sequence length to multiple of chunk_size
        chunk_size = self.config.chunk_size
        pad_len = (chunk_size - T % chunk_size) % chunk_size
        if pad_len > 0:
            idx = F.pad(idx, (0, pad_len), value=0)
            if targets is not None:
                targets = F.pad(targets, (0, pad_len), value=-1)

        # Embed bytes
        x = self.wte(idx)
        x = x.to(COMPUTE_DTYPE)
        x = norm(x)

        # Mamba2 layers
        for layer in self.layers:
            x = layer(x)

        x = norm(x)

        # Remove padding before computing logits
        if pad_len > 0:
            x = x[:, :T]
            if targets is not None:
                targets = targets[:, :T]

        # LM head with logit softcap
        softcap = 15
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1,
                                   reduction=loss_reduction)
        return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Autoregressive byte generation using step-based recurrence.
        tokens: list of ints (byte values 0-255)
        Yields one byte (int) at a time.
        """
        device = self.get_device()
        cfg = self.config
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)

        # Allocate per-layer recurrent state
        conv_dim = cfg.d_inner + 2 * cfg.ngroups * cfg.d_state
        conv_states = [torch.zeros(1, conv_dim, cfg.d_conv, device=device,
                                   dtype=COMPUTE_DTYPE) for _ in range(cfg.n_layer)]
        ssm_states = [torch.zeros(1, cfg.nheads, cfg.headdim, cfg.d_state, device=device,
                                  dtype=COMPUTE_DTYPE) for _ in range(cfg.n_layer)]

        # Prefill: run all prompt bytes through step-by-step
        for tok in tokens:
            x = self.wte(torch.tensor([[tok]], device=device, dtype=torch.long))
            x = x.squeeze(1).to(COMPUTE_DTYPE)
            x = norm(x)
            for i, layer in enumerate(self.layers):
                x = layer.step(x, conv_states[i], ssm_states[i])
            x = norm(x)

        # Generate new bytes
        logits = self.lm_head(x)[:, :cfg.vocab_size].float()
        softcap = 15
        logits = softcap * torch.tanh(logits / softcap)

        for _ in range(max_tokens):
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)

            token = next_id.item()
            yield token

            x = self.wte(next_id).squeeze(1).to(COMPUTE_DTYPE)
            x = norm(x)
            for i, layer in enumerate(self.layers):
                x = layer.step(x, conv_states[i], ssm_states[i])
            x = norm(x)
            logits = self.lm_head(x)[:, :cfg.vocab_size].float()
            logits = softcap * torch.tanh(logits / softcap)

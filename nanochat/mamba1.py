"""
Mamba1 byte-level language model (MambaByte style).
Notable features:
- byte-level: vocab_size=256, no tokenizer needed
- selective state space model (S6) with input-dependent dynamics
- O(1) inference memory per layer (no KV cache growth)
- Triton selective scan kernel (pure PyTorch fallback available)
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
    from nanochat.triton_scan import selective_scan_triton
    HAS_TRITON_SCAN = True
except ImportError:
    HAS_TRITON_SCAN = False


# ─── Config ─────────────────────────────────────────────────────────────────

@dataclass
class Mamba1Config:
    sequence_len: int = 8192  # byte-level needs longer sequences (~4x token-level)
    vocab_size: int = 256     # raw UTF-8 bytes
    n_layer: int = 24          # ~2x GPT layers (Mamba layers are ~half the FLOPs each)
    n_embd: int = 768         # d_model
    expand: int = 2           # d_inner = expand * n_embd
    d_state: int = 16         # SSM state dimension (N in the paper)
    d_conv: int = 4           # causal conv1d kernel width

    def __post_init__(self):
        self.d_inner = self.expand * self.n_embd
        self.dt_rank = math.ceil(self.n_embd / 16)


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


# ─── Selective Scan ─────────────────────────────────────────────────────────
# The scan computes: h[t] = dA[t] * h[t-1] + dBx[t], y[t] = C[t] . h[t]
# where dA = exp(dt * A) and dBx = dt * B * x (ZOH discretization).
#
# Two backends:
#   selective_scan     — O(log L) depth parallel scan (training default)
#   selective_scan_ref — O(L) sequential scan (reference / testing)
#
# Both will be replaced by a Triton kernel for cross-platform GPU performance.

class ParallelScan(torch.autograd.Function):
    """
    Blelloch parallel prefix scan for the SSM recurrence.
    Operates in O(L) work and O(log L) sequential depth.

    Inputs/outputs are (B, L, D, N) — batch, time, d_inner, d_state.
    Internally transposes to (B, D, L, N) for contiguous time-axis access.
    """

    @staticmethod
    def _scan(A, X):
        """In-place Blelloch scan. Modifies X to hold prefix sums."""
        B, D, L, N = A.shape
        num_steps = int(math.log2(L))

        # ── Up-sweep (reduction) ──
        Aa, Xa = A, X
        for _ in range(num_steps):
            T = 2 * (Xa.size(2) // 2)
            Aa = Aa[:, :, :T].view(B, D, T // 2, 2, N)
            Xa = Xa[:, :, :T].view(B, D, T // 2, 2, N)
            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])
            Aa = Aa[:, :, :, 1]
            Xa = Xa[:, :, :, 1]

        # ── Down-sweep ──
        for k in range(num_steps - 1, -1, -1):
            Aa = A[:, :, 2**k - 1::2**k]
            Xa = X[:, :, 2**k - 1::2**k]
            T = 2 * (Xa.size(2) // 2)
            if T < Xa.size(2):
                Xa[:, :, -1].add_(Aa[:, :, -1].mul(Xa[:, :, -2]))
                Aa[:, :, -1].mul_(Aa[:, :, -2])
            Aa = Aa[:, :, :T].view(B, D, T // 2, 2, N)
            Xa = Xa[:, :, :T].view(B, D, T // 2, 2, N)
            Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
            Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])

    @staticmethod
    def forward(ctx, A_in, X_in):
        # Clone + transpose to (B, D, L, N) for contiguous scan
        A = A_in.clone().transpose(1, 2)
        X = X_in.clone().transpose(1, 2)
        ParallelScan._scan(A, X)
        ctx.save_for_backward(A_in, X)
        return X.transpose(1, 2)

    @staticmethod
    def backward(ctx, grad_output_in):
        A_in, X = ctx.saved_tensors
        # Reverse scan for gradient backpropagation
        A = A_in.clone().transpose(1, 2)  # (B, D, L, N)
        A = torch.cat([A[:, :, :1], A[:, :, 1:].flip(2)], dim=2)
        grad = grad_output_in.transpose(1, 2).flip(2).clone()
        ParallelScan._scan(A, grad)
        grad = grad.flip(2)
        # Gradient w.r.t. A: dA[t] = grad[t] * X[t-1]
        Q = torch.zeros_like(X)
        Q[:, :, 1:].add_(X[:, :, :-1] * grad[:, :, 1:])
        return Q.transpose(1, 2), grad.transpose(1, 2)

pscan = ParallelScan.apply


def selective_scan(x, dt, A, B, C, D):
    """
    Mamba1 selective scan using parallel prefix scan.

    Args:
        x:  (B, L, D_inner) — input signal (after conv1d + silu)
        dt: (B, L, D_inner) — timestep delta (after softplus)
        A:  (D_inner, N)    — state transition matrix (negative)
        B:  (B, L, N)       — input-dependent input matrix
        C:  (B, L, N)       — input-dependent output matrix
        D:  (D_inner,)      — skip connection
    Returns:
        y:  (B, L, D_inner)
    """
    # Discretize (ZOH): dA = exp(dt * A), dBx = dt * B * x
    dA = torch.exp(dt.unsqueeze(-1) * A)                        # (B, L, D, N)
    dBx = dt.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)   # (B, L, D, N)

    # Parallel scan: h[t] = dA[t] * h[t-1] + dBx[t]
    h = pscan(dA, dBx)  # (B, L, D, N)

    # Output: y[t] = C[t] . h[t] + D * x[t]
    y = (h * C.unsqueeze(2)).sum(-1)  # (B, L, D)
    y = y + D * x
    return y


def selective_scan_ref(x, dt, A, B, C, D):
    """Reference sequential scan for testing. Same API as selective_scan."""
    B_batch, L, d_inner = x.shape
    dA = torch.exp(dt.unsqueeze(-1) * A)                        # (B, L, D, N)
    dBx = dt.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)   # (B, L, D, N)
    h = torch.zeros(B_batch, d_inner, A.shape[1], device=x.device, dtype=x.dtype)
    ys = []
    for t in range(L):
        h = dA[:, t] * h + dBx[:, t]
        y_t = (h * C[:, t].unsqueeze(1)).sum(-1)
        ys.append(y_t)
    y = torch.stack(ys, dim=1)
    return y + D * x


# ─── Model Components ──────────────────────────────────────────────────────

class Mamba1Mixer(nn.Module):
    """
    Mamba1 selective SSM mixer.

    Signal flow:
        input ── in_proj ──┬── conv1d → silu ── x_proj ──┬── dt_proj → softplus → dt
                            │                              ├── B (input-dependent)
                            │                              └── C (input-dependent)
                            │     selective_scan(x, dt, A, B, C, D) → y
                            └── z ── silu(z) ─────────────────── × y
                                                                  │
                                                              out_proj → output
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        d, d_inner = config.n_embd, config.d_inner
        dt_rank, d_state, d_conv = config.dt_rank, config.d_state, config.d_conv

        # Projections
        self.in_proj = Linear(d, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv,
                                groups=d_inner, padding=d_conv - 1, bias=True)
        self.x_proj = Linear(d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = Linear(dt_rank, d_inner, bias=True)
        self.out_proj = Linear(d_inner, d, bias=False)

        # SSM parameters (initialized in Mamba1.init_weights)
        self.A_log = nn.Parameter(torch.empty(d_inner, d_state))
        self.D = nn.Parameter(torch.empty(d_inner))

    def forward(self, x):
        """x: (B, L, D) → (B, L, D)"""
        _, L, _ = x.shape

        # Dual branch: x (signal) + z (gate)
        xz = self.in_proj(x)               # (B, L, 2*D_inner)
        x, z = xz.chunk(2, dim=-1)         # each (B, L, D_inner)

        # Causal depthwise conv1d + SiLU activation
        x = x.transpose(1, 2)  # (B, D_inner, L)
        x = F.conv1d(x, self.conv1d.weight.to(dtype=x.dtype),
                     self.conv1d.bias.to(dtype=x.dtype),
                     padding=self.conv1d.padding, groups=self.conv1d.groups)[:, :, :L]
        x = x.transpose(1, 2)  # (B, L, D_inner)
        x = F.silu(x)

        # Compute input-dependent SSM parameters
        x_dbl = self.x_proj(x)             # (B, L, dt_rank + 2*N)
        dt_rank = self.dt_proj.in_features
        d_state = (x_dbl.shape[-1] - dt_rank) // 2
        dt, B_ssm, C = x_dbl.split([dt_rank, d_state, d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))   # (B, L, D_inner)

        # SSM scan — Triton kernel on GPU, pure PyTorch fallback on CPU
        A = -torch.exp(self.A_log.float())  # (D_inner, N) — always negative
        scan_fn = selective_scan_triton if (HAS_TRITON_SCAN and x.is_cuda) else selective_scan
        y = scan_fn(x, dt, A, B_ssm, C, self.D.float())

        # Gate with SiLU(z) and project back to model dim
        return self.out_proj(y * F.silu(z))

    def step(self, x, conv_state, ssm_state):
        """
        Single-step recurrent inference. O(1) per step (no recomputation).
        x: (B, D) → (B, D)
        conv_state: (B, D_inner, d_conv) — mutable, updated in place
        ssm_state:  (B, D_inner, N)      — mutable, updated in place
        """
        xz = self.in_proj(x)               # (B, 2*D_inner)
        x, z = xz.chunk(2, dim=-1)         # each (B, D_inner)

        # Conv step: shift state, insert new input, apply conv (cast to match forward() dtype)
        conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
        conv_state[:, :, -1] = x
        conv_w = self.conv1d.weight.squeeze(1).to(dtype=x.dtype)
        x = (conv_state * conv_w).sum(-1)
        if self.conv1d.bias is not None:
            x = x + self.conv1d.bias.to(dtype=x.dtype)
        x = F.silu(x)

        # SSM step
        x_dbl = self.x_proj(x)
        dt_rank = self.dt_proj.in_features
        d_state = (x_dbl.shape[-1] - dt_rank) // 2
        dt, B_ssm, C = x_dbl.split([dt_rank, d_state, d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))   # (B, D_inner)

        A = -torch.exp(self.A_log.float())
        dA = torch.exp(dt.unsqueeze(-1) * A)           # (B, D_inner, N)
        dB = dt.unsqueeze(-1) * B_ssm.unsqueeze(1)     # (B, D_inner, N)
        ssm_state.copy_(ssm_state * dA + x.unsqueeze(-1) * dB)
        y = (ssm_state.to(x.dtype) * C.unsqueeze(1)).sum(-1) + self.D.to(x.dtype) * x

        return self.out_proj(y * F.silu(z))


class Mamba1Block(nn.Module):
    """Pre-norm residual block: x + Mixer(norm(x))"""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.mixer = Mamba1Mixer(config, layer_idx)

    def forward(self, x):
        return x + self.mixer(norm(x))

    def step(self, x, conv_state, ssm_state):
        return x + self.mixer.step(norm(x), conv_state, ssm_state)


# ─── Full Model ─────────────────────────────────────────────────────────────

class Mamba1(nn.Module):
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
        self.layers = nn.ModuleList([Mamba1Block(config, i) for i in range(config.n_layer)])
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
        s = 3**0.5 * d**-0.5       # uniform bound matching normal std at d_model scale
        s_inner = 3**0.5 * d_inner**-0.5

        for block in self.layers:
            m = block.mixer

            # Projection weights
            nn.init.uniform_(m.in_proj.weight, -s, s)
            nn.init.uniform_(m.x_proj.weight, -s_inner, s_inner)
            nn.init.zeros_(m.out_proj.weight)       # zero init: residual starts neutral

            # dt_proj: Mamba-specific initialization
            dt_init_std = cfg.dt_rank**-0.5
            nn.init.uniform_(m.dt_proj.weight, -dt_init_std, dt_init_std)
            # bias: inverse-softplus of uniform in [dt_min, dt_max] log-space
            dt = torch.exp(
                torch.rand(d_inner, device=m.dt_proj.bias.device)
                * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
            ).clamp(min=1e-4)
            m.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inv_softplus

            # A_log: S4D real initialization — log(1), log(2), ..., log(N)
            A = torch.arange(1, cfg.d_state + 1, dtype=torch.float32,
                             device=m.A_log.device).repeat(d_inner, 1)
            m.A_log.copy_(torch.log(A))

            # D: skip connection, initialize to 1
            m.D.fill_(1.0)

            # conv1d: kaiming uniform (matches nn.Conv1d default reset_parameters)
            fan_in = m.conv1d.in_channels // m.conv1d.groups * m.conv1d.kernel_size[0]
            nn.init.kaiming_uniform_(m.conv1d.weight, a=math.sqrt(5))
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(m.conv1d.bias, -bound, bound)

        # Cast embedding to compute dtype (bf16 usually). Skip for fp16 (GradScaler needs fp32).
        if COMPUTE_DTYPE != torch.float16:
            self.wte.to(dtype=COMPUTE_DTYPE)

    def get_device(self):
        return self.wte.weight.device

    def estimate_flops(self):
        """
        Estimated FLOPs per token (forward + backward).
        Each matmul parameter contributes 6 FLOPs (2 fwd, 4 bwd).
        Non-matmul ops (SSM scan, conv1d, elementwise) are small and excluded.
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params
        nparams_exclude = self.wte.weight.numel()  # embedding lookup
        for block in self.layers:
            m = block.mixer
            nparams_exclude += (m.A_log.numel() + m.D.numel()
                                + m.conv1d.weight.numel() + m.conv1d.bias.numel()
                                + m.dt_proj.bias.numel())
        return 6 * (nparams - nparams_exclude)

    def num_scaling_params(self):
        """Parameter counts by group for scaling law analysis."""
        wte = self.wte.weight.numel()
        lm_head = self.lm_head.weight.numel()
        layer_matmul = sum(
            m.mixer.in_proj.weight.numel() + m.mixer.x_proj.weight.numel()
            + m.mixer.dt_proj.weight.numel() + m.mixer.out_proj.weight.numel()
            for m in self.layers)
        layer_ssm = sum(
            m.mixer.A_log.numel() + m.mixer.D.numel() + m.mixer.dt_proj.bias.numel()
            for m in self.layers)
        layer_conv = sum(
            m.mixer.conv1d.weight.numel() + m.mixer.conv1d.bias.numel()
            for m in self.layers)
        total = wte + lm_head + layer_matmul + layer_ssm + layer_conv
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'lm_head': lm_head,
            'transformer_matrices': layer_matmul,
            'layer_ssm': layer_ssm,
            'layer_conv': layer_conv,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.008, embedding_lr=0.3, matrix_lr=0.02,
                        ssm_lr=0.001, conv_lr=0.01, weight_decay=0.0):
        """
        Set up MuonAdamW optimizer with Mamba-appropriate parameter groups.

        - Muon for 2D weight matrices (in_proj, x_proj, dt_proj, out_proj)
        - AdamW for embeddings, LM head, SSM params (A_log, D), conv, dt bias
        """
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # ── Collect parameter groups ──
        matrix_params = []   # 2D projection weights → Muon
        ssm_params = []      # A_log, D → AdamW, no weight decay
        conv_params = []     # conv1d weight + bias → AdamW
        dt_bias_params = []  # dt_proj.bias → AdamW, no weight decay

        for block in self.layers:
            m = block.mixer
            matrix_params.extend([m.in_proj.weight, m.x_proj.weight,
                                  m.dt_proj.weight, m.out_proj.weight])
            ssm_params.extend([m.A_log, m.D])
            conv_params.extend([m.conv1d.weight, m.conv1d.bias])
            dt_bias_params.append(m.dt_proj.bias)

        embedding_params = list(self.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())

        # Verify completeness
        n_total = (len(matrix_params) + len(ssm_params) + len(conv_params)
                   + len(dt_bias_params) + len(embedding_params) + len(lm_head_params))
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
            dict(kind='adamw', params=dt_bias_params, lr=ssm_lr,
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

        # Embed bytes
        x = self.wte(idx)                   # (B, T, D)
        x = x.to(COMPUTE_DTYPE)
        x = norm(x)                          # post-embedding norm

        # Mamba layers
        for layer in self.layers:
            x = layer(x)

        x = norm(x)                          # final norm

        # LM head with logit softcap
        softcap = 15
        logits = self.lm_head(x)                        # (B, T, padded_vocab)
        logits = logits[..., :self.config.vocab_size]    # crop padding
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)  # smooth clamp to [-15, 15]

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
        conv_states = [torch.zeros(1, cfg.d_inner, cfg.d_conv, device=device,
                                   dtype=COMPUTE_DTYPE) for _ in range(cfg.n_layer)]
        ssm_states = [torch.zeros(1, cfg.d_inner, cfg.d_state, device=device,
                                  dtype=COMPUTE_DTYPE) for _ in range(cfg.n_layer)]

        # Prefill: run all prompt bytes through step-by-step
        for tok in tokens:
            x = self.wte(torch.tensor([[tok]], device=device, dtype=torch.long))
            x = x.squeeze(1).to(COMPUTE_DTYPE)  # (1, D)
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

            # Step the model with the new byte
            x = self.wte(next_id).squeeze(1).to(COMPUTE_DTYPE)
            x = norm(x)
            for i, layer in enumerate(self.layers):
                x = layer.step(x, conv_states[i], ssm_states[i])
            x = norm(x)
            logits = self.lm_head(x)[:, :cfg.vocab_size].float()
            logits = softcap * torch.tanh(logits / softcap)

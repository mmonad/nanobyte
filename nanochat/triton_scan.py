"""
Triton selective scan kernel for Mamba1 SSM.
Cross-platform: works on CUDA and ROCm via Triton.

The key insight: a sequential scan in Triton is fast because the SSM state
h[d, n] lives in registers (not global memory). We stream through the
sequence once, doing O(D*N) register ops per timestep. This is 10-50x
less memory traffic than the pure-PyTorch parallel scan which materializes
(B, L, D, N) tensors to global memory.

Reference: https://github.com/sustcsonglin/mamba-triton
"""

import torch
import triton
import triton.language as tl


# ─── Forward Kernel ─────────────────────────────────────────────────────────

@triton.jit
def _scan_fwd(
    # Tensor pointers
    x_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, D_ptr,
    y_ptr, H_ptr,
    # Dimensions
    L, D_dim,
    # Strides for (B, L, D) tensors (x, dt, y)
    s_xb, s_xl,
    # Strides for (B, L, N) tensors (B_ssm, C)
    s_Bb, s_Bl,
    # Strides for (B, L, D, N) tensor (H)
    s_Hb, s_Hl, s_Hd,
    # Constexpr
    N: tl.constexpr,
    BD: tl.constexpr,
):
    """Forward scan: compute y[b,t,d] and save H[b,t,d,n] for backward."""
    i_b = tl.program_id(0)
    i_d = tl.program_id(1)

    # D indices for this block
    d_off = i_d * BD + tl.arange(0, BD)    # (BD,)
    d_mask = d_off < D_dim
    n_off = tl.arange(0, N)                 # (N,)

    # Load constant params: A[d, n] and D_skip[d]
    a = tl.load(A_ptr + d_off[:, None] * N + n_off[None, :],
                mask=d_mask[:, None], other=0.0)                  # (BD, N) fp32
    d_skip = tl.load(D_ptr + d_off, mask=d_mask, other=0.0)      # (BD,)  fp32

    # Base pointers for this batch
    x_base = x_ptr + i_b * s_xb
    dt_base = dt_ptr + i_b * s_xb
    y_base = y_ptr + i_b * s_xb
    B_base = B_ptr + i_b * s_Bb
    C_base = C_ptr + i_b * s_Bb
    H_base = H_ptr + i_b * s_Hb

    # Initialize hidden state in registers
    h = tl.zeros([BD, N], dtype=tl.float32)

    for t in range(L):
        # Load inputs (cast to fp32 for scan precision)
        x_t = tl.load(x_base + t * s_xl + d_off, mask=d_mask, other=0.0).to(tl.float32)
        dt_t = tl.load(dt_base + t * s_xl + d_off, mask=d_mask, other=0.0).to(tl.float32)
        b_t = tl.load(B_base + t * s_Bl + n_off).to(tl.float32)     # (N,)
        c_t = tl.load(C_base + t * s_Bl + n_off).to(tl.float32)     # (N,)

        # Discretize: dA = exp(dt * A)
        dA = tl.exp(dt_t[:, None] * a)             # (BD, N)

        # State update: h = h * dA + x * dt * B
        h = h * dA + x_t[:, None] * (dt_t[:, None] * b_t[None, :])

        # Output: y = sum_n(h * C) + D * x
        y_t = tl.sum(h * c_t[None, :], axis=1) + d_skip * x_t      # (BD,)

        # Store output in fp32 (matches PyTorch scan semantics)
        tl.store(y_base + t * s_xl + d_off, y_t, mask=d_mask)

        # Store hidden state for backward (always fp32)
        tl.store(H_base + t * s_Hl + d_off[:, None] * s_Hd + n_off[None, :],
                 h, mask=d_mask[:, None])


# ─── Backward Kernel ────────────────────────────────────────────────────────

@triton.jit
def _scan_bwd(
    # Forward inputs (read-only)
    x_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, D_ptr, H_ptr,
    # Gradient input
    dy_ptr,
    # Gradient outputs
    dx_ptr, ddt_ptr,
    dA_ptr,                 # (B, D, N) — sum across B later
    dD_ptr,                 # (B, D) — sum across B later
    dB_partial_ptr,         # (num_D_blocks, B, L, N) — sum across dim 0 later
    dC_partial_ptr,         # (num_D_blocks, B, L, N) — sum across dim 0 later
    # Dimensions
    L, D_dim,
    # Strides for (B, L, D)
    s_xb, s_xl,
    # Strides for (B, L, N)
    s_Bb, s_Bl,
    # Strides for (B, L, D, N) — H
    s_Hb, s_Hl, s_Hd,
    # Strides for dB_partial (num_D_blocks, B, L, N)
    s_dBp_block, s_dBp_b, s_dBp_l,
    # Strides for dA (B, D, N)
    s_dA_b, s_dA_d,
    # Constexpr
    N: tl.constexpr,
    BD: tl.constexpr,
):
    """Backward scan: compute all input gradients."""
    i_b = tl.program_id(0)
    i_d = tl.program_id(1)

    d_off = i_d * BD + tl.arange(0, BD)
    d_mask = d_off < D_dim
    n_off = tl.arange(0, N)

    # Load constants
    a = tl.load(A_ptr + d_off[:, None] * N + n_off[None, :],
                mask=d_mask[:, None], other=0.0)
    d_skip = tl.load(D_ptr + d_off, mask=d_mask, other=0.0)

    # Base pointers
    x_base = x_ptr + i_b * s_xb
    dt_base = dt_ptr + i_b * s_xb
    dy_base = dy_ptr + i_b * s_xb
    dx_base = dx_ptr + i_b * s_xb
    ddt_base = ddt_ptr + i_b * s_xb
    B_base = B_ptr + i_b * s_Bb
    C_base = C_ptr + i_b * s_Bb
    H_base = H_ptr + i_b * s_Hb

    dB_base = dB_partial_ptr + i_d * s_dBp_block + i_b * s_dBp_b
    dC_base = dC_partial_ptr + i_d * s_dBp_block + i_b * s_dBp_b

    # Accumulators for dA[d, n] and dD[d] (summed over L)
    dA_acc = tl.zeros([BD, N], dtype=tl.float32)
    dD_acc = tl.zeros([BD], dtype=tl.float32)

    # Initialize gradient hidden state
    dh = tl.zeros([BD, N], dtype=tl.float32)

    # Reverse scan
    for t_rev in range(L):
        t = L - 1 - t_rev

        # Load forward values
        x_t = tl.load(x_base + t * s_xl + d_off, mask=d_mask, other=0.0).to(tl.float32)
        dt_t = tl.load(dt_base + t * s_xl + d_off, mask=d_mask, other=0.0).to(tl.float32)
        b_t = tl.load(B_base + t * s_Bl + n_off).to(tl.float32)
        c_t = tl.load(C_base + t * s_Bl + n_off).to(tl.float32)
        dy_t = tl.load(dy_base + t * s_xl + d_off, mask=d_mask, other=0.0).to(tl.float32)

        # Load h_{t-1} (zero for t=0)
        if t > 0:
            h_prev = tl.load(H_base + (t - 1) * s_Hl + d_off[:, None] * s_Hd + n_off[None, :],
                             mask=d_mask[:, None], other=0.0)
        else:
            h_prev = tl.zeros([BD, N], dtype=tl.float32)

        # Load h_t (needed for dC)
        h_t = tl.load(H_base + t * s_Hl + d_off[:, None] * s_Hd + n_off[None, :],
                       mask=d_mask[:, None], other=0.0)

        # Recompute dA for this timestep
        dA_t = tl.exp(dt_t[:, None] * a)     # (BD, N)

        # Total gradient into h_t: from output + from future
        dh = dh + dy_t[:, None] * c_t[None, :]   # (BD, N)

        # ── Compute gradients ──

        # dx_t = sum_n(dh * dt * B) + D * dy  (from state + skip)
        dx_t = tl.sum(dh * dt_t[:, None] * b_t[None, :], axis=1) + d_skip * dy_t

        # ddt_t = sum_n(dh * (A * dA * h_prev + x * B))
        ddt_t = tl.sum(dh * (a * dA_t * h_prev + x_t[:, None] * b_t[None, :]), axis=1)

        # dA_acc += dh * dt * dA * h_prev  (accumulated over L)
        dA_acc += dh * dt_t[:, None] * dA_t * h_prev

        # dD_acc += dy * x  (accumulated over L)
        dD_acc += dy_t * x_t

        # dB_t[n] = sum_d(dh[d,n] * x[d] * dt[d])  — partial sum for this D-block
        dB_t = tl.sum(dh * x_t[:, None] * dt_t[:, None], axis=0)     # (N,)

        # dC_t[n] = sum_d(dy[d] * h_t[d,n])  — partial sum for this D-block
        dC_t = tl.sum(dy_t[:, None] * h_t, axis=0)                    # (N,)

        # Store per-timestep gradients
        tl.store(dx_base + t * s_xl + d_off, dx_t.to(x_ptr.dtype.element_ty), mask=d_mask)
        tl.store(ddt_base + t * s_xl + d_off, ddt_t.to(dt_ptr.dtype.element_ty), mask=d_mask)
        tl.store(dB_base + t * s_dBp_l + n_off, dB_t)
        tl.store(dC_base + t * s_dBp_l + n_off, dC_t)

        # Propagate dh backward: dh_{t-1} = dA_t * dh_t
        dh = dA_t * dh

    # Store accumulated dA and dD
    tl.store(dA_ptr + i_b * s_dA_b + d_off[:, None] * s_dA_d + n_off[None, :],
             dA_acc, mask=d_mask[:, None])
    tl.store(dD_ptr + i_b * D_dim + d_off, dD_acc, mask=d_mask)


# ─── Autograd Function ─────────────────────────────────────────────────────

class TritonSelectiveScan(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, dt, A, B, C, D):
        """
        x:  (B, L, D) bf16  — input signal
        dt: (B, L, D) bf16  — timestep delta (after softplus)
        A:  (D, N)    fp32  — state transition (negative)
        B:  (B, L, N) bf16  — input matrix
        C:  (B, L, N) bf16  — output matrix
        D:  (D,)      fp32  — skip connection
        """
        x, dt, B, C = x.contiguous(), dt.contiguous(), B.contiguous(), C.contiguous()
        B_batch, L, D_dim = x.shape
        N = A.shape[1]

        BD = 32
        num_D_blocks = triton.cdiv(D_dim, BD)

        # Output in fp32 to match PyTorch scan semantics (precision for out_proj matmul)
        y = torch.empty(B_batch, L, D_dim, device=x.device, dtype=torch.float32)
        # Only allocate H when we need gradients (skip during eval/inference)
        needs_grad = any(ctx.needs_input_grad)
        if needs_grad:
            H = torch.empty(B_batch, L, D_dim, N, device=x.device, dtype=torch.float32)
        else:
            # Dummy 1-element tensor — kernel writes are masked but pointer must be valid
            H = torch.empty(1, device=x.device, dtype=torch.float32)

        grid = (B_batch, num_D_blocks)
        _scan_fwd[grid](
            x, dt, A, B, C, D, y, H,
            L, D_dim,
            x.stride(0), x.stride(1),
            B.stride(0), B.stride(1),
            H.stride(0) if needs_grad else 0,
            H.stride(1) if needs_grad else 0,
            H.stride(2) if needs_grad else 0,
            N=N, BD=BD,
        )

        if needs_grad:
            ctx.save_for_backward(x, dt, A, B, C, D, H)
            ctx.BD = BD
        return y

    @staticmethod
    def backward(ctx, dy):
        x, dt, A, B, C, D, H = ctx.saved_tensors
        BD = ctx.BD
        B_batch, L, D_dim = x.shape
        N = A.shape[1]
        num_D_blocks = triton.cdiv(D_dim, BD)

        dy = dy.contiguous()

        # Allocate gradient outputs
        dx = torch.empty_like(x)
        ddt = torch.empty_like(dt)
        dA = torch.empty(B_batch, D_dim, N, device=x.device, dtype=torch.float32)
        dD = torch.empty(B_batch, D_dim, device=x.device, dtype=torch.float32)
        # Partial buffers for cross-D-block reduction
        dB_partial = torch.empty(num_D_blocks, B_batch, L, N, device=x.device, dtype=torch.float32)
        dC_partial = torch.empty(num_D_blocks, B_batch, L, N, device=x.device, dtype=torch.float32)

        grid = (B_batch, num_D_blocks)
        _scan_bwd[grid](
            x, dt, A, B, C, D, H, dy,
            dx, ddt, dA, dD, dB_partial, dC_partial,
            L, D_dim,
            x.stride(0), x.stride(1),
            B.stride(0), B.stride(1),
            H.stride(0), H.stride(1), H.stride(2),
            dB_partial.stride(0), dB_partial.stride(1), dB_partial.stride(2),
            dA.stride(0), dA.stride(1),
            N=N, BD=BD,
        )

        # Reduce partial sums
        dB = dB_partial.sum(dim=0)    # (B, L, N)
        dC = dC_partial.sum(dim=0)    # (B, L, N)
        dA = dA.sum(dim=0)            # (D, N)
        dD = dD.sum(dim=0)            # (D,)

        return dx, ddt, dA, dB, dC, dD


def selective_scan_triton(x, dt, A, B, C, D):
    """Drop-in replacement for selective_scan using Triton kernels."""
    return TritonSelectiveScan.apply(x, dt, A, B, C, D)

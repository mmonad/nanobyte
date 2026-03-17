"""
Train Mamba1 byte-level model. From root directory:

    python -m scripts.mamba_train
    python -m scripts.mamba_train --depth=4 --num-iterations=100  # quick test
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import time
import math
import argparse

import torch

from nanochat.mamba1 import Mamba1, Mamba1Config
from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON

# ─── CLI ────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Pretrain Mamba1 byte-level model")
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty=autodetect)")
# Model
parser.add_argument("--depth", type=int, default=12, help="number of Mamba layers")
parser.add_argument("--aspect-ratio", type=int, default=64, help="n_embd = depth * aspect_ratio")
parser.add_argument("--max-seq-len", type=int, default=1024, help="byte sequence length")
# Training
parser.add_argument("--num-iterations", type=int, default=500, help="number of optimization steps")
parser.add_argument("--device-batch-size", type=int, default=8, help="per-device batch size (sequences)")
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in bytes (-1=device_batch_size*max_seq_len)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="Muon LR for projection weights")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="AdamW LR for embedding")
parser.add_argument("--unembedding-lr", type=float, default=0.008, help="AdamW LR for lm_head")
parser.add_argument("--ssm-lr", type=float, default=0.001, help="AdamW LR for SSM params (A_log, D)")
parser.add_argument("--weight-decay", type=float, default=0.1, help="weight decay for Muon")
parser.add_argument("--warmup-steps", type=int, default=20, help="LR warmup steps")
parser.add_argument("--warmdown-ratio", type=float, default=0.5, help="fraction of training for LR warmdown")
# Data
parser.add_argument("--data", type=str, default="", help="path to text file for training (empty=synthetic)")
# Display
parser.add_argument("--log-every", type=int, default=10, help="print every N steps")
parser.add_argument("--val-every", type=int, default=50, help="validate every N steps (-1=disable)")
args = parser.parse_args()

# ─── Setup ──────────────────────────────────────────────────────────────────

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

if device_type == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_name)
    print0(f"GPU: {gpu_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

# ─── Data ───────────────────────────────────────────────────────────────────

def load_byte_data(path, device):
    """Load a text file as a flat byte tensor."""
    with open(path, 'rb') as f:
        data = f.read()
    return torch.tensor(list(data), dtype=torch.long, device=device)

def generate_synthetic_data(n_bytes, device):
    """Generate synthetic byte data for smoke-testing the training loop."""
    text = ("The quick brown fox jumps over the lazy dog. "
            "Pack my box with five dozen liquor jugs. "
            "How vexingly quick daft zebras jump! "
            "Sphinx of black quartz, judge my vow. ") * (n_bytes // 160 + 1)
    data = list(text.encode('utf-8'))[:n_bytes]
    return torch.tensor(data, dtype=torch.long, device=device)

total_bytes_needed = args.device_batch_size * args.max_seq_len * (args.num_iterations + 100)
total_bytes_needed = max(total_bytes_needed, 1_000_000)

if args.data:
    print0(f"Loading data from {args.data}")
    all_bytes = load_byte_data(args.data, device)
else:
    print0(f"Using synthetic data ({total_bytes_needed:,} bytes)")
    all_bytes = generate_synthetic_data(total_bytes_needed, device)

split = int(len(all_bytes) * 0.9)
train_bytes = all_bytes[:split]
val_bytes = all_bytes[split:]
print0(f"Data: {len(train_bytes):,} train bytes, {len(val_bytes):,} val bytes")

def get_batch(split_data, batch_size, seq_len):
    """Sample a random batch of byte sequences."""
    ix = torch.randint(0, len(split_data) - seq_len, (batch_size,))
    x = torch.stack([split_data[i:i+seq_len] for i in ix])
    y = torch.stack([split_data[i+1:i+seq_len+1] for i in ix])
    return x, y

# ─── Model ──────────────────────────────────────────────────────────────────

n_embd = args.depth * args.aspect_ratio
config = Mamba1Config(
    sequence_len=args.max_seq_len,
    n_layer=args.depth,
    n_embd=n_embd,
)
print0(f"Model: n_layer={config.n_layer}, n_embd={config.n_embd}, "
       f"d_inner={config.d_inner}, dt_rank={config.dt_rank}, seq_len={config.sequence_len}")

with torch.device("meta"):
    model = Mamba1(config)

model.to_empty(device=device)
model.init_weights()

param_counts = model.num_scaling_params()
print0(f"Parameters: {param_counts['total']:,}")
num_flops_per_token = model.estimate_flops()
print0(f"FLOPs/token: {num_flops_per_token:e}")

# ─── Optimizer ──────────────────────────────────────────────────────────────

optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    ssm_lr=args.ssm_lr,
    weight_decay=args.weight_decay,
)

scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None

# ─── Compile ────────────────────────────────────────────────────────────────

orig_model = model
model = torch.compile(model, dynamic=False)

# ─── Batch / grad accum ────────────────────────────────────────────────────

total_batch_size = args.total_batch_size
if total_batch_size == -1:
    total_batch_size = args.device_batch_size * args.max_seq_len
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
assert total_batch_size % tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // tokens_per_fwdbwd
print0(f"Batch: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,} bytes/step, "
       f"grad_accum={grad_accum_steps}")

# ─── LR schedule ────────────────────────────────────────────────────────────

num_iterations = args.num_iterations

def get_lr_multiplier(it):
    warmup = args.warmup_steps
    warmdown = round(args.warmdown_ratio * num_iterations)
    if it < warmup:
        return (it + 1) / warmup
    elif it <= num_iterations - warmdown:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown
        return progress * 1.0 + (1 - progress) * 0.05

# ─── Validation ─────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, n_batches=10):
    model_was_training = model.training
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(val_bytes, args.device_batch_size, args.max_seq_len)
        loss = model(x, y)
        losses.append(loss.item())
    if model_was_training:
        model.train()
    avg_loss = sum(losses) / len(losses)
    bpb = avg_loss / math.log(2)
    return avg_loss, bpb

# ─── Training loop ──────────────────────────────────────────────────────────

print0(f"\nTraining for {num_iterations} steps...")
print0(f"{'step':>6} {'loss':>8} {'bpb':>8} {'dt_ms':>8} {'tok/s':>10} {'mfu%':>6}")
print0("-" * 55)

smooth_loss = 0
total_training_time = 0

for step in range(num_iterations + 1):
    last_step = step == num_iterations

    # Periodic validation
    if args.val_every > 0 and (last_step or step % args.val_every == 0):
        val_loss, val_bpb = validate(model)
        print0(f"  EVAL | val_loss={val_loss:.4f} val_bpb={val_bpb:.4f}")

    if last_step:
        break

    # ── Training step ──
    synchronize()
    t0 = time.time()

    for micro_step in range(grad_accum_steps):
        x, y = get_batch(train_bytes, args.device_batch_size, args.max_seq_len)
        loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

    # Update LR
    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm

    # Step optimizer
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()
    synchronize()
    t1 = time.time()
    dt = t1 - t0

    # ── Logging ──
    ema = 0.9
    smooth_loss = ema * smooth_loss + (1 - ema) * train_loss_f
    debiased = smooth_loss / (1 - ema**(step + 1))
    bpb = debiased / math.log(2)

    if step > 5:
        total_training_time += dt

    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / gpu_peak_flops

    if step % args.log_every == 0 or step < 5:
        print0(f"{step:6d} {debiased:8.4f} {bpb:8.4f} {dt*1000:8.1f} {tok_per_sec:10,} {mfu:6.2f}")

    # GC
    if step == 0:
        gc.collect()
        gc.disable()
    elif step % 5000 == 0:
        gc.collect()

# ─── Done ───────────────────────────────────────────────────────────────────

print0(f"\nPeak memory: {get_max_memory() / 1024**2:.0f} MiB")
print0(f"Total training time: {total_training_time:.1f}s ({total_training_time/60:.1f}m)")
compute_cleanup()

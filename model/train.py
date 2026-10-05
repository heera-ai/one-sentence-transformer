"""Train the tiny GPT.

Every step: take a batch of text windows, ask the model to predict each next
token, measure how surprised it was (the loss), and nudge every weight a
little in the direction that would have made it less surprised.

    uv run python train.py                      # the model shipped with the site
    uv run python train.py --n_layer 6 --d_model 192 --n_head 6
"""

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig

OUT = Path(__file__).parent / "out"
# Snapshots for the "training time machine" chapter (the final step is added below).
SNAPSHOT_STEPS = [0, 50, 200, 500, 1000, 3000, 10000]


def get_batch(data: np.ndarray, batch_size: int, block_size: int, device: str):
    """Random windows of text. The target is the same window shifted by one token."""
    starts = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[s : s + block_size] for s in starts]).astype(np.int64)
    y = np.stack([data[s + 1 : s + 1 + block_size] for s in starts]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def lr_at(step: int, args) -> float:
    """Linear warm-up, then cosine decay."""
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    progress = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, data, args, device, batches=20) -> float:
    model.eval()
    losses = [model(*get_batch(data, args.batch_size, model.cfg.block_size, device))[1].item() for _ in range(batches)]
    model.train()
    return sum(losses) / len(losses)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--block_size", type=int, default=64)
    p.add_argument("--vocab_size", type=int, default=4096)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    train_data = np.memmap(OUT / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(OUT / "val.bin", dtype=np.uint16, mode="r")

    cfg = GPTConfig(args.vocab_size, args.block_size, args.n_layer, args.n_head, args.d_model)
    model = GPT(cfg).to(device)
    print(f"{model.num_params():,} parameters on {device}")

    # Weight decay on the matrices only, not on biases or layer-norm gains.
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
        betas=(0.9, 0.95),
    )

    ckpt_dir = OUT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    snapshots = set(SNAPSHOT_STEPS) | {args.steps}
    log = []
    t0 = time.time()

    for step in range(args.steps + 1):
        if step in snapshots:
            torch.save({"config": asdict(cfg), "state": model.state_dict(), "step": step}, ckpt_dir / f"step_{step}.pt")

        if step % args.eval_every == 0 or step in snapshots:
            val = estimate_loss(model, val_data, args, device)
            train = estimate_loss(model, train_data, args, device)
            log.append({"step": step, "train": round(train, 4), "val": round(val, 4), "lr": lr_at(step, args), "secs": round(time.time() - t0)})
            print(f"step {step:6d}  train {train:.3f}  val {val:.3f}  {time.time() - t0:5.0f}s", flush=True)
            (OUT / "trainlog.json").write_text(json.dumps(log))

        if step == args.steps:
            break

        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, args)
        x, y = get_batch(train_data, args.batch_size, cfg.block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


if __name__ == "__main__":
    main()

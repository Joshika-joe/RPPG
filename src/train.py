"""
Train an rPPG model on the per-subject window pipeline.

    python -m src.train --model v2 --run v2_newpipe
    python -m src.train --model v2 --run v2_newpipe --resume        # continue from last.pth

Every run writes to results/<run>/:
    config.json      all arguments (so src.evaluate can reload the run)
    history.json     per-epoch train loss + val loss / Pearson / HR MAE
    best.pth         weights of the best epoch by --select-on
    last.pth         full state (model, optimizer, epoch, counters) for --resume
    loss_curve.png
"""

import argparse
import json
import os
import time
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from . import config
from .dataset import make_eval_loader, make_train_loader
from .evaluate import evaluate
from .models import MODEL_NAMES, build_model, count_params, run_model

# Which validation metric picks the best epoch, and whether lower is better.
SELECT_CRITERIA = {
    "mse": ("val_mse", True),
    "pearson": ("val_pearson", False),
    "hr_mae": ("val_hr_mae", True),
}


class NegPearsonLoss(nn.Module):
    """1 - Pearson r per window, averaged. Scale-invariant: rewards waveform shape, not amplitude."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = pred - pred.mean(dim=1, keepdim=True)
        t = target - target.mean(dim=1, keepdim=True)
        r = (p * t).sum(dim=1) / (torch.sqrt((p * p).sum(dim=1) * (t * t).sum(dim=1)) + 1e-8)
        return (1.0 - r).mean()


class PearsonMSELoss(nn.Module):
    """(1 - r) + mse_weight * MSE. The small MSE term keeps the output on the target's scale."""

    def __init__(self, mse_weight: float = config.PEARSON_MSE_WEIGHT):
        super().__init__()
        self.pearson = NegPearsonLoss()
        self.mse = nn.MSELoss()
        self.mse_weight = mse_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.pearson(pred, target) + self.mse_weight * self.mse(pred, target)


def build_loss(name: str) -> nn.Module:
    if name == "mse":
        return nn.MSELoss()
    if name == "pearson":
        return PearsonMSELoss()
    raise ValueError(f"Unknown loss {name!r}")


def build_optimizer(name: str, params, lr: float, wd: float) -> torch.optim.Optimizer:
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer {name!r}")


def build_scheduler(name: str, optimizer, total_steps: int, warmup_steps: int, min_ratio: float = 0.02):
    """Per-step LR multiplier: linear warmup then cosine decay to min_ratio (or constant)."""
    if name == "none":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    if name == "cosine":
        import math

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    raise ValueError(f"Unknown scheduler {name!r}")


def save_history_plot(history: Dict, path: str) -> None:
    ep = history["epoch"]
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(ep, history["train_loss"], label="train loss")
    ax1.plot(ep, history["val_mse"], label="val MSE")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("MSE")
    ax1.grid(True, alpha=0.4)
    ax2 = ax1.twinx()
    ax2.plot(ep, history["val_pearson"], "g--", label="val Pearson r")
    ax2.plot(ep, [h / 10 for h in history["val_hr_mae"]], "r:", label="val HR MAE / 10 bpm")
    ax2.set_ylabel("Pearson r  /  HR MAE (÷10)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def train(args) -> Dict:
    config.set_seed(args.seed)
    device = config.get_device()
    torch.set_num_threads(args.threads or torch.get_num_threads())

    run_dir = os.path.join(config.RESULTS_DIR, args.run)
    os.makedirs(run_dir, exist_ok=True)
    best_path = os.path.join(run_dir, "best.pth")
    last_path = os.path.join(run_dir, "last.pth")

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # ---------------------------------------------------------------- data
    train_loader = make_train_loader(
        batch_size=args.batch_size,
        stride=args.stride,
        normalize=args.normalize,
        augment=not args.no_augment,
        windows_per_epoch=args.windows_per_epoch,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    val_loader = make_eval_loader(
        config.VAL_SUBJECTS,
        batch_size=args.batch_size,
        stride=config.EVAL_STRIDE,
        normalize=args.normalize,
    )
    n_train_total = len(train_loader.dataset)
    n_train_epoch = len(train_loader.sampler)

    # ---------------------------------------------------------------- model
    model = build_model(args.model).to(device)
    criterion = build_loss(args.loss)
    optimizer = build_optimizer(args.optimizer, model.parameters(), args.lr, args.wd)
    steps_per_epoch = len(train_loader)
    scheduler = build_scheduler(
        args.scheduler, optimizer, total_steps=args.epochs * steps_per_epoch,
        warmup_steps=args.warmup_epochs * steps_per_epoch,
    )

    metric_key, lower_is_better = SELECT_CRITERIA[args.select_on]
    best_score = float("inf") if lower_is_better else -float("inf")
    patience_counter = 0
    start_epoch = 1
    history: Dict[str, list] = {
        "epoch": [], "train_loss": [], "val_mse": [], "val_pearson": [], "val_hr_mae": [], "lr": [], "epoch_time_s": [],
    }

    if args.resume and os.path.exists(last_path):
        state = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        best_score = state["best_score"]
        patience_counter = state["patience_counter"]
        start_epoch = state["epoch"] + 1
        history = state["history"]
        history.setdefault("lr", [])  # runs recorded before lr logging existed
        print(f"Resumed from epoch {state['epoch']} (best {metric_key}={best_score:.4f})")

    print("\n=== TRAINING ===")
    print(f"  run            {run_dir}")
    print(f"  model          {args.model} ({count_params(model):,} params)")
    print(f"  device         {device}  (threads {torch.get_num_threads()})")
    print(f"  train windows  {n_train_total} total (stride {args.stride}), {n_train_epoch} per epoch")
    print(f"  val windows    {len(val_loader.dataset)} (stride {config.EVAL_STRIDE})")
    print(f"  normalize      {args.normalize}   augment {not args.no_augment}")
    print(f"  loss {args.loss}  optimizer {args.optimizer}  lr {args.lr}  wd {args.wd}  batch {args.batch_size}")
    print(f"  scheduler {args.scheduler} (warmup {args.warmup_epochs} ep)  grad clip {args.grad_clip}")
    print(f"  epochs {args.epochs}  patience {args.patience}  select on {metric_key}\n")

    # ---------------------------------------------------------------- loop
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        running = 0.0
        n_batches = 0
        for x, y, meta in train_loader:
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(run_model(model, x, meta, device), y)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            running += loss.item()
            n_batches += 1
        train_loss = running / max(n_batches, 1)
        current_lr = optimizer.param_groups[0]["lr"]

        val_metrics, _, _, _ = evaluate(model, val_loader, device)
        s = val_metrics["summary"]
        scores = {"val_mse": s["mse"], "val_pearson": s["pearson_mean"], "val_hr_mae": s["hr_mae_bpm"]}
        epoch_time = time.time() - t0

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_mse"].append(scores["val_mse"])
        history["val_pearson"].append(scores["val_pearson"])
        history["val_hr_mae"].append(scores["val_hr_mae"])
        history["lr"].append(current_lr)
        history["epoch_time_s"].append(epoch_time)

        score = scores[metric_key]
        improved = score < best_score if lower_is_better else score > best_score
        if improved:
            best_score = score
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:03d}/{args.epochs} | train {train_loss:.4f} | "
            f"val MSE {scores['val_mse']:.4f}  r {scores['val_pearson']:.3f}  "
            f"HR MAE {scores['val_hr_mae']:.2f} bpm  amp {s['amplitude_ratio_mean']:.2f} | "
            f"lr {current_lr:.1e} | {epoch_time / 60:.1f} min{'  * best' if improved else ''}",
            flush=True,
        )

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_score": best_score,
                "patience_counter": patience_counter,
                "history": history,
            },
            last_path,
        )
        with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        save_history_plot(history, os.path.join(run_dir, "loss_curve.png"))

        if patience_counter >= args.patience:
            print(f"\nEarly stopping: no {metric_key} improvement for {args.patience} epochs.")
            break

    print(f"\nBest {metric_key}: {best_score:.4f}  ->  {best_path}")
    return history


def main() -> None:
    p = argparse.ArgumentParser(description="Train an rPPG model")
    p.add_argument("--model", default="v3", choices=MODEL_NAMES)
    p.add_argument("--run", required=True, help="run name -> results/<run>/")
    p.add_argument("--loss", default="mse", choices=["mse", "pearson"], help="pearson = (1-r) + %.2f*MSE" % config.PEARSON_MSE_WEIGHT)
    p.add_argument("--optimizer", default="adam", choices=["adam", "adamw"])
    p.add_argument("--scheduler", default="none", choices=["none", "cosine"])
    p.add_argument("--warmup-epochs", type=int, default=0)
    p.add_argument("--grad-clip", type=float, default=0.0, help="max grad norm (0 = off)")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--patience", type=int, default=config.PATIENCE)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--wd", type=float, default=config.WEIGHT_DECAY)
    p.add_argument("--stride", type=int, default=config.TRAIN_STRIDE)
    p.add_argument("--windows-per-epoch", type=int, default=config.WINDOWS_PER_EPOCH)
    p.add_argument("--normalize", default=config.NORMALIZE, choices=["raw", "temporal"])
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--select-on", default="mse", choices=list(SELECT_CRITERIA))
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()

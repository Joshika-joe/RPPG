"""
Evaluate a trained model on a split and write metrics + diagnostic plots.

    # A run directory produced by src.train (reads its config.json + best.pth)
    python -m src.evaluate --run results/v2_newpipe --split test

    # An explicit checkpoint (e.g. the legacy v2 weights, trained on raw pixels)
    python -m src.evaluate --checkpoint best_rppg_model_v2.pth --model v2 --normalize raw --split val

    # Sanity check: legacy pre-cut .npy validation windows from the original pipeline
    python -m src.evaluate --checkpoint best_rppg_model_v2.pth --model v2 --legacy-val

Outputs in --out (default results/<run-or-checkpoint>_<split>/):
    metrics.json      summary + per-subject + per-window metrics
    per_window.csv
    waveforms.png     best / median / worst Pearson windows
    hr_scatter.png    predicted vs reference HR
    bland_altman.png
"""

import argparse
import csv
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config
from .dataset import LegacyValDataset, make_eval_loader
from .metrics import compute_metrics, format_summary
from .models import build_model, count_params, load_weights


# ==============================================================================
# INFERENCE
# ==============================================================================
@torch.no_grad()
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, Dict[str, List]]:
    """Runs the model over a loader. Returns (preds, targets, info) with info columns as lists."""
    model.eval()
    preds, targets = [], []
    info: Dict[str, List] = {"subject": [], "start": [], "fps": [], "hr_ref": []}
    for x, y, meta in loader:
        out = model(x.to(device)).cpu().numpy()
        preds.append(out)
        targets.append(y.numpy())
        info["subject"].extend(list(meta["subject"]))
        info["start"].extend(meta["start"].tolist())
        info["fps"].extend(meta["fps"].tolist())
        info["hr_ref"].extend(meta["hr_ref"].tolist())
    return np.concatenate(preds), np.concatenate(targets), info


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    bandpass: bool = False,
) -> Tuple[Dict, np.ndarray, np.ndarray, Dict[str, List]]:
    preds, targets, info = predict(model, loader, device)
    metrics = compute_metrics(
        preds,
        targets,
        fps=np.array(info["fps"]),
        hr_ref=np.array(info["hr_ref"]),
        subjects=info["subject"],
        bandpass=bandpass,
    )
    return metrics, preds, targets, info


# ==============================================================================
# PLOTS
# ==============================================================================
def plot_waveforms(preds, targets, metrics, info, path: str, n: int = 3) -> None:
    r = np.array(metrics["per_window"]["pearson"])
    order = np.argsort(r)
    picks = [order[-1], order[len(order) // 2], order[0]][:n]
    labels = ["best", "median", "worst"]
    fig, axes = plt.subplots(len(picks), 1, figsize=(12, 3.2 * len(picks)))
    axes = np.atleast_1d(axes)
    for ax, idx, lab in zip(axes, picks, labels):
        ax.plot(targets[idx], label="Ground truth BVP", linewidth=1.4)
        ax.plot(preds[idx], "--", label="Predicted BVP", linewidth=1.4)
        ax.set_title(
            f"{lab}: {info['subject'][idx]} @ frame {info['start'][idx]}  |  "
            f"r = {r[idx]:.2f}, HR pred {metrics['per_window']['hr_pred_bpm'][idx]:.1f} / "
            f"gt {metrics['per_window']['hr_gt_bpm'][idx]:.1f} bpm"
        )
        ax.set_xlabel("Frame")
        ax.set_ylabel("Normalized BVP")
        ax.grid(True, alpha=0.4)
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_hr_scatter(metrics, path: str) -> None:
    gt = np.array(metrics["per_window"]["hr_gt_bpm"])
    pr = np.array(metrics["per_window"]["hr_pred_bpm"])
    lo, hi = np.nanmin([gt.min(), pr.min()]) - 5, np.nanmax([gt.max(), pr.max()]) + 5
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(gt, pr, s=18, alpha=0.7)
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="identity")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Reference HR (bpm)")
    ax.set_ylabel("Predicted HR (bpm)")
    s = metrics["summary"]
    ax.set_title(f"HR MAE {s['hr_mae_bpm']:.2f} bpm, RMSE {s['hr_rmse_bpm']:.2f} bpm")
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_bland_altman(metrics, path: str) -> None:
    gt = np.array(metrics["per_window"]["hr_gt_bpm"])
    pr = np.array(metrics["per_window"]["hr_pred_bpm"])
    mean = (gt + pr) / 2
    diff = pr - gt
    md, sd = np.nanmean(diff), np.nanstd(diff)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(mean, diff, s=18, alpha=0.7)
    ax.axhline(md, color="k", linewidth=1, label=f"mean {md:+.2f}")
    ax.axhline(md + 1.96 * sd, color="r", linestyle="--", linewidth=1, label=f"±1.96 SD ({1.96 * sd:.2f})")
    ax.axhline(md - 1.96 * sd, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("Mean of reference and predicted HR (bpm)")
    ax.set_ylabel("Predicted − reference HR (bpm)")
    ax.set_title("Bland–Altman")
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ==============================================================================
# OUTPUT
# ==============================================================================
def write_outputs(out_dir: str, metrics: Dict, preds, targets, info, extra: Dict) -> None:
    os.makedirs(out_dir, exist_ok=True)

    payload = {"run": extra, "summary": metrics["summary"], "per_subject": metrics.get("per_subject", {})}
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    pw = metrics["per_window"]
    cols = ["subject", "start", "fps"] + [k for k in pw if k not in ("subject", "fps")]
    with open(os.path.join(out_dir, "per_window.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(len(preds)):
            row = [info["subject"][i], info["start"][i], f"{info['fps'][i]:.3f}"]
            row += [f"{pw[k][i]:.5f}" if isinstance(pw[k][i], float) else pw[k][i] for k in cols[3:]]
            w.writerow(row)

    plot_waveforms(preds, targets, metrics, info, os.path.join(out_dir, "waveforms.png"))
    plot_hr_scatter(metrics, os.path.join(out_dir, "hr_scatter.png"))
    plot_bland_altman(metrics, os.path.join(out_dir, "bland_altman.png"))


def print_report(title: str, metrics: Dict) -> None:
    print(f"\n=== {title} ===")
    print(format_summary(metrics["summary"]))
    if "per_subject" in metrics:
        print("\n  per subject:")
        print(f"  {'subject':<10} {'n':>3} {'r':>6} {'HR MAE':>7} {'SNR':>6} {'amp':>5}")
        for s, m in metrics["per_subject"].items():
            print(
                f"  {s:<10} {m['n_windows']:>3} {m['pearson_mean']:>6.3f} "
                f"{m['hr_mae_bpm']:>7.2f} {m['snr_db_mean']:>6.2f} {m['amplitude_ratio_mean']:>5.2f}"
            )


# ==============================================================================
# CLI
# ==============================================================================
def resolve_run(args) -> Tuple[str, str, str, str]:
    """Returns (checkpoint_path, model_name, normalize, run_label)."""
    if args.run:
        run_dir = args.run if os.path.isabs(args.run) else os.path.join(config.PROJECT_ROOT, args.run)
        with open(os.path.join(run_dir, "config.json"), "r", encoding="utf-8") as f:
            run_cfg = json.load(f)
        ckpt = os.path.join(run_dir, "best.pth")
        return ckpt, run_cfg["model"], run_cfg["normalize"], os.path.basename(run_dir.rstrip("/\\"))
    if not args.checkpoint or not args.model:
        raise SystemExit("Provide --run DIR, or --checkpoint PATH with --model NAME")
    ckpt = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(config.PROJECT_ROOT, args.checkpoint)
    label = os.path.splitext(os.path.basename(ckpt))[0]
    return ckpt, args.model, args.normalize, label


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate an rPPG model")
    p.add_argument("--run", help="run directory from src.train (uses config.json + best.pth)")
    p.add_argument("--checkpoint", help="explicit weights file")
    p.add_argument("--model", choices=["v2"], help="model name (with --checkpoint)")
    p.add_argument("--normalize", default="raw", choices=["raw", "temporal"], help="input normalization (with --checkpoint)")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--stride", type=int, default=config.EVAL_STRIDE)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--bandpass", action="store_true", help="band-pass predictions before metrics")
    p.add_argument("--legacy-val", action="store_true", help="use the original pipeline's rppg_X_val.npy")
    p.add_argument("--out", help="output directory (default results/<label>_<split>)")
    args = p.parse_args()

    device = config.get_device()
    ckpt, model_name, normalize, label = resolve_run(args)

    model = build_model(model_name)
    load_weights(model, ckpt, device)
    print(f"Model {model_name} ({count_params(model):,} params) <- {ckpt}")
    print(f"Device: {device}")

    if args.legacy_val:
        ds = LegacyValDataset()
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        split_label = "legacyval"
        print(f"Legacy validation windows: {len(ds)} (raw pixels, fps assumed 30)")
    else:
        subjects = config.SPLITS[args.split]
        loader = make_eval_loader(subjects, batch_size=args.batch_size, stride=args.stride, normalize=normalize)
        split_label = args.split
        print(f"Split {args.split}: {len(subjects)} subjects, {len(loader.dataset)} windows "
              f"(stride {args.stride}, normalize={normalize})")

    t0 = time.time()
    metrics, preds, targets, info = evaluate(model, loader, device, bandpass=args.bandpass)
    print(f"Inference: {time.time() - t0:.1f}s")

    print_report(f"{label} / {split_label}", metrics)

    out_dir = args.out or os.path.join(config.RESULTS_DIR, f"{label}_{split_label}")
    extra = {
        "checkpoint": os.path.relpath(ckpt, config.PROJECT_ROOT),
        "model": model_name,
        "normalize": normalize,
        "split": split_label,
        "stride": args.stride,
        "bandpass": args.bandpass,
        "params": count_params(model),
    }
    write_outputs(out_dir, metrics, preds, targets, info, extra)
    print(f"\nWrote metrics + plots to {os.path.relpath(out_dir, config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

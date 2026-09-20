"""
Evaluate a trained model on a split and write metrics + diagnostic plots.

    # A run directory produced by src.train (reads its config.json + best.pth)
    python -m src.evaluate --run results/v2_newpipe --split test

    # An explicit checkpoint (e.g. the legacy v2 weights, trained on raw pixels)
    python -m src.evaluate --checkpoint best_rppg_model_v2.pth --model v2 --normalize raw --split val

    # Sanity check: legacy pre-cut .npy validation windows from the original pipeline
    python -m src.evaluate --checkpoint best_rppg_model_v2.pth --model v2 --legacy-val

    # Continuous inference: slide with --stride, overlap-add into one BVP per subject,
    # score on non-overlapping --segment-frames segments (adds continuous.png)
    python -m src.evaluate --run results/v3_pearson --split test --continuous --stride 30 --segment-frames 300 --bandpass

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
from .inference import predict_continuous, segment_signals
from .metrics import bandpass_filter, compute_metrics, format_summary, pearson_per_window
from .models import MODEL_NAMES, build_model, count_params, load_weights, run_model


# ==============================================================================
# INFERENCE
# ==============================================================================
@torch.no_grad()
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, Dict[str, List]]:
    """
    Runs the model over a loader. Returns (preds, targets, info) with info columns as lists.
    For models exposing `last_attention`, info also gets per-window "attention" (h, w) maps
    and "mean_frame" (3, H, W) arrays.
    """
    model.eval()
    preds, targets = [], []
    info: Dict[str, List] = {"subject": [], "start": [], "fps": [], "hr_ref": []}
    has_attention = hasattr(model, "last_attention")
    if has_attention:
        info["attention"], info["mean_frame"] = [], []
    for x, y, meta in loader:
        out = run_model(model, x, meta, device).cpu().numpy()
        preds.append(out)
        targets.append(y.numpy())
        info["subject"].extend(list(meta["subject"]))
        info["start"].extend(meta["start"].tolist())
        info["fps"].extend(meta["fps"].tolist())
        info["hr_ref"].extend(meta["hr_ref"].tolist())
        if has_attention and model.last_attention is not None:
            info["attention"].extend(model.last_attention[:, 0].cpu().numpy())
            info["mean_frame"].extend(meta["mean_frame"].numpy())
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


def plot_attention(info, path: str, n_subjects: int = 4) -> None:
    """Per-subject mean attention mask (upsampled) over the subject's mean face."""
    subjects = sorted(set(info["subject"]))[:n_subjects]
    fig, axes = plt.subplots(2, len(subjects), figsize=(3.2 * len(subjects), 6.4), squeeze=False)
    for col, s in enumerate(subjects):
        idx = [i for i, name in enumerate(info["subject"]) if name == s]
        face = np.mean([info["mean_frame"][i] for i in idx], axis=0)[::-1].transpose(1, 2, 0)  # BGR -> RGB
        att = np.mean([info["attention"][i] for i in idx], axis=0)
        axes[0, col].imshow(np.clip(face, 0, 1))
        axes[0, col].set_title(f"{s}: mean face")
        axes[1, col].imshow(np.clip(face, 0, 1))
        h, w = face.shape[:2]
        axes[1, col].imshow(att, cmap="jet", alpha=0.45, extent=(0, w, h, 0), interpolation="bilinear")
        axes[1, col].set_title(f"attention (min {att.min():.2f}, max {att.max():.2f})")
        for ax in axes[:, col]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_continuous(signals: Dict, path: str, seconds: float = 20.0, bandpass: bool = False) -> None:
    """One row per subject: continuous prediction vs ground truth for the first `seconds`."""
    names = list(signals)
    fig, axes = plt.subplots(len(names), 1, figsize=(13, 2.6 * len(names)), squeeze=False)
    for ax, name in zip(axes[:, 0], names):
        sig = signals[name]
        fps = sig["fps"]
        n = min(len(sig["pred"]), int(seconds * fps))
        pred, gt = sig["pred"][:n], sig["gt"][:n]
        if bandpass:
            pred, gt = bandpass_filter(pred, fps), bandpass_filter(gt, fps)
        t = np.arange(n) / fps
        ax.plot(t, (gt - gt.mean()) / (gt.std() + 1e-8), linewidth=1.2, label="Ground truth BVP")
        ax.plot(t, (pred - pred.mean()) / (pred.std() + 1e-8), "--", linewidth=1.2, label="Prediction (overlap-add)")
        ax.set_title(f"{name}  (whole-recording r = {sig['subject_r']:.2f}, {sig['n_windows']} windows)")
        ax.set_xlabel("seconds")
        ax.grid(True, alpha=0.4)
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def evaluate_continuous(
    model: nn.Module,
    subjects,
    device: torch.device,
    stride: int,
    normalize: str,
    segment_frames: int,
    bandpass: bool,
    batch_size: int,
):
    """Overlap-add inference over whole recordings, then segment-level metrics."""
    signals = predict_continuous(model, subjects, stride, normalize, device, batch_size)
    for sig in signals.values():  # whole-recording correlation (band-passed on both sides)
        pred, gt = sig["pred"], sig["gt"]
        if bandpass:
            pred, gt = bandpass_filter(pred, sig["fps"]), bandpass_filter(gt, sig["fps"])
        sig["subject_r"] = float(pearson_per_window(pred[None], gt[None])[0])
    preds, targets, fps, subj, hr_ref = segment_signals(signals, segment_frames)
    metrics = compute_metrics(preds, targets, fps=fps, hr_ref=hr_ref, subjects=subj, bandpass=bandpass)
    for name, sig in signals.items():
        metrics["per_subject"][name]["whole_recording_r"] = sig["subject_r"]
        metrics["per_subject"][name]["n_windows"] = sig["n_windows"]
    metrics["summary"]["whole_recording_r_mean"] = float(np.mean([s["subject_r"] for s in signals.values()]))
    info = {
        "subject": subj,
        "start": [i * segment_frames for i in range(len(subj))],
        "fps": fps.tolist(),
        "hr_ref": hr_ref.tolist(),
    }
    return metrics, preds, targets, info, signals


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
    if info.get("attention"):
        plot_attention(info, os.path.join(out_dir, "attention.png"))


def print_report(title: str, metrics: Dict) -> None:
    print(f"\n=== {title} ===")
    print(format_summary(metrics["summary"]))
    if "whole_recording_r_mean" in metrics["summary"]:
        print(f"  {'whole-recording r':<20} {metrics['summary']['whole_recording_r_mean']:.3f}")
    if "per_subject" in metrics:
        has_wr = any("whole_recording_r" in m for m in metrics["per_subject"].values())
        print("\n  per subject:")
        print(f"  {'subject':<10} {'n':>3} {'r':>6} {'HR MAE':>7} {'SNR':>6} {'amp':>5}" + ("  rec-r" if has_wr else ""))
        for s, m in metrics["per_subject"].items():
            line = (
                f"  {s:<10} {m['n_windows']:>3} {m['pearson_mean']:>6.3f} "
                f"{m['hr_mae_bpm']:>7.2f} {m['snr_db_mean']:>6.2f} {m['amplitude_ratio_mean']:>5.2f}"
            )
            if "whole_recording_r" in m:
                line += f"  {m['whole_recording_r']:>5.2f}"
            print(line)


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
    p.add_argument("--model", choices=MODEL_NAMES, help="model name (with --checkpoint)")
    p.add_argument("--normalize", default="raw", choices=["raw", "temporal"], help="input normalization (with --checkpoint)")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--stride", type=int, default=config.EVAL_STRIDE)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--bandpass", action="store_true", help="band-pass predictions before metrics")
    p.add_argument("--legacy-val", action="store_true", help="use the original pipeline's rppg_X_val.npy")
    p.add_argument("--continuous", action="store_true", help="overlap-add sliding inference per subject (uses --stride)")
    p.add_argument("--segment-frames", type=int, default=config.SEGMENT_FRAMES, help="segment length for continuous metrics")
    p.add_argument("--out", help="output directory (default results/<label>_<split>)")
    args = p.parse_args()

    device = config.get_device()
    ckpt, model_name, normalize, label = resolve_run(args)

    model = build_model(model_name)
    load_weights(model, ckpt, device)
    print(f"Model {model_name} ({count_params(model):,} params) <- {ckpt}")
    print(f"Device: {device}")

    signals = None
    t0 = time.time()
    if args.continuous:
        subjects = config.SPLITS[args.split]
        split_label = f"{args.split}_continuous"
        print(f"Split {args.split}: {len(subjects)} subjects, continuous overlap-add "
              f"(stride {args.stride}, segments of {args.segment_frames} frames, normalize={normalize})")
        metrics, preds, targets, info, signals = evaluate_continuous(
            model, subjects, device, stride=args.stride, normalize=normalize,
            segment_frames=args.segment_frames, bandpass=args.bandpass, batch_size=args.batch_size,
        )
    elif args.legacy_val:
        ds = LegacyValDataset()
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        split_label = "legacyval"
        print(f"Legacy validation windows: {len(ds)} (raw pixels, fps assumed 30)")
        metrics, preds, targets, info = evaluate(model, loader, device, bandpass=args.bandpass)
    else:
        subjects = config.SPLITS[args.split]
        loader = make_eval_loader(subjects, batch_size=args.batch_size, stride=args.stride, normalize=normalize)
        split_label = args.split
        print(f"Split {args.split}: {len(subjects)} subjects, {len(loader.dataset)} windows "
              f"(stride {args.stride}, normalize={normalize})")
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
        "continuous": args.continuous,
        "segment_frames": args.segment_frames if args.continuous else None,
        "params": count_params(model),
    }
    write_outputs(out_dir, metrics, preds, targets, info, extra)
    if signals is not None:
        plot_continuous(signals, os.path.join(out_dir, "continuous.png"), bandpass=args.bandpass)
    print(f"\nWrote metrics + plots to {os.path.relpath(out_dir, config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

"""
Zero-shot evaluation of an existing checkpoint on an adapted external dataset.

Nothing is retrained, fine-tuned, selected or tuned on the target data. The checkpoint, the
10-second estimation window, the FFT estimator, the 0.7-4 Hz band, the band-pass, the
overlap-add reconstruction and the quality-gate thresholds are all the ones already fixed on
UBFC-rPPG. Only the input format is adapted, by the dataset's own preprocessing module.

    python -m src.eval_cross --run results/v3_pearson \
        --subject-root checkpoints/subjects_ubfc_phys --label ubfc_phys

Uncertainty is bootstrapped over PARTICIPANTS. UBFC-Phys records each participant three
times, so recordings are clustered by participant; treating 168 recordings as independent
would understate the interval badly.
"""

import argparse
import csv
import json
import os
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import config
from .compare_windows import GATE, gate_decision
from .failure_analysis import bootstrap_subject_ci
from .hr_estimators import estimate_hr_detail, peak_margin_db
from .inference import predict_continuous
from .metrics import bandpass_filter, pearson_per_window, snr_db
from .models import build_model, load_weights
from .preprocessing import load_subject, subject_is_processed


def discover(subject_root: str) -> List[str]:
    if not os.path.isdir(subject_root):
        raise SystemExit(
            f"No preprocessed store at {subject_root!r}. Adapt the dataset first, e.g.\n"
            f"  python -m src.ubfc_phys --root /path/to/UBFC-Phys"
        )
    names = sorted(n for n in os.listdir(subject_root)
                   if subject_is_processed(n, subject_root))
    if not names:
        raise SystemExit(f"{subject_root!r} contains no complete recordings.")
    return names


def best_lag_correlation(pred: np.ndarray, gt: np.ndarray, fps: float,
                         max_lag_s: float = 1.0) -> Dict:
    """
    Waveform correlation at zero lag, and the best correlation within +/- max_lag_s.

    Reported as a DIAGNOSTIC only. A wrist reference arrives after the facial pulse and may
    carry an unknown constant offset from the camera, so zero-lag correlation understates
    waveform agreement. The best-lag value is never used to select anything.
    """
    n = int(max_lag_s * fps)
    zero = float(pearson_per_window(pred[None], gt[None])[0])
    best, best_lag = zero, 0
    for lag in range(-n, n + 1):
        if lag < 0:
            a, b = pred[-lag:], gt[: len(gt) + lag]
        elif lag > 0:
            a, b = pred[: len(pred) - lag], gt[lag:]
        else:
            continue
        if len(a) < fps * 2:
            continue
        r = float(pearson_per_window(a[None], b[None])[0])
        if r > best:
            best, best_lag = r, lag
    return {"r_zero_lag": zero, "r_best_lag": best, "best_lag_s": best_lag / fps}


def main() -> None:
    p = argparse.ArgumentParser(description="Zero-shot cross-dataset evaluation")
    p.add_argument("--run", default="results/v3_pearson")
    p.add_argument("--subject-root", required=True)
    p.add_argument("--label", default="cross")
    p.add_argument("--stride", type=int, default=config.CONTINUOUS_STRIDE)
    p.add_argument("--segment-frames", type=int, default=config.SEGMENT_FRAMES,
                   help="frames per HR segment; at the target's own fps this is ~10 s")
    p.add_argument("--segment-seconds", type=float, default=None,
                   help="use a fixed duration in seconds instead of a frame count")
    p.add_argument("--method", default="fft")
    p.add_argument("--recordings", nargs="*", default=None)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    device = config.get_device()
    run_dir = os.path.join(config.PROJECT_ROOT, args.run)
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    model = build_model(cfg["model"])
    load_weights(model, os.path.join(run_dir, "best.pth"), device)

    root = args.subject_root if os.path.isabs(args.subject_root) else os.path.join(
        config.PROJECT_ROOT, args.subject_root)
    names = args.recordings or discover(root)
    metas = {n: load_subject(n, root)["meta"] for n in names}
    participants = sorted({m.get("participant", m["subject"]) for m in metas.values()})
    datasets = sorted({m.get("dataset", "unknown") for m in metas.values()})

    print(f"Zero-shot evaluation - NO retraining, fine-tuning or target-side tuning")
    print(f"  checkpoint      {args.run} ({cfg['model']}, normalize={cfg['normalize']})")
    print(f"  target dataset  {', '.join(datasets)}")
    print(f"  recordings      {len(names)} from {len(participants)} participant(s)")
    print(f"  estimator       {args.method}, band {config.HR_BAND_HZ[0]}-{config.HR_BAND_HZ[1]} Hz")
    print(f"  gate            {GATE}")
    fps_values = sorted({round(m["fps"], 1) for m in metas.values()})
    print(f"  target fps      {fps_values} (model input resampled to its trained 150 frames)")

    signals = predict_continuous(model, names, args.stride, cfg["normalize"], device,
                                 subject_root=root)

    records: List[Dict] = []
    skipped: Dict[str, str] = {}
    for name in names:
        if name not in signals:
            skipped[name] = "too short for one model window"
            continue
        sig = signals[name]
        fps = sig["fps"]
        meta = metas[name]
        n = int(round(args.segment_seconds * fps)) if args.segment_seconds else args.segment_frames
        pred, gt = sig["pred"], sig["gt"]
        for end in range(n, len(pred) + 1, n):
            pf = bandpass_filter(pred[end - n : end], fps)
            tf = bandpass_filter(gt[end - n : end], fps)
            if pf.std() == 0 or tf.std() == 0:
                continue
            d_p = estimate_hr_detail(pf, fps, method=args.method)
            d_r = estimate_hr_detail(tf, fps, method=args.method)
            if not np.isfinite(d_r["hr_bpm"]):
                skipped.setdefault(name, "reference unreadable in >=1 segment")
                continue
            hr_p, hr_r = d_p["hr_bpm"], d_r["hr_bpm"]
            cross = estimate_hr_detail(pf, fps, method="autocorr")["hr_bpm"]
            ind = {"hr": hr_p, "snr": snr_db(pf, hr_p, fps),
                   "margin": peak_margin_db(d_p), "cross": cross}
            reasons = gate_decision(ind)
            corr = best_lag_correlation(pf, tf, fps)
            records.append({
                "recording": name,
                "participant": meta.get("participant", name),
                "task": meta.get("task_name", meta.get("task", "")),
                "fps": float(fps),
                "end_s": float(end / fps),
                "hr_pred": float(hr_p), "hr_ref": float(hr_r),
                "err": float(abs(hr_p - hr_r)) if np.isfinite(hr_p) else float("nan"),
                "snr": float(ind["snr"]), "margin": float(ind["margin"]),
                "cross": float(cross),
                "r_zero_lag": corr["r_zero_lag"], "r_best_lag": corr["r_best_lag"],
                "best_lag_s": corr["best_lag_s"],
                "accepted": not reasons, "reasons": reasons,
            })

    if not records:
        raise SystemExit("No evaluable segments were produced.")

    err = np.array([r["err"] for r in records], dtype=float)
    ok = np.array([r["accepted"] for r in records], dtype=bool)
    valid = np.isfinite(err)
    acc = err[ok & valid]

    by_part: Dict[str, List[float]] = {}
    for r in records:
        if np.isfinite(r["err"]):
            by_part.setdefault(r["participant"], []).append(r["err"])
    part_mae = {k: float(np.mean(v)) for k, v in by_part.items()}
    ci = bootstrap_subject_ci(part_mae)

    by_task: Dict[str, List[float]] = {}
    for r in records:
        if np.isfinite(r["err"]):
            by_task.setdefault(r["task"] or "all", []).append(r["err"])

    reasons: Dict[str, int] = {}
    for r in records:
        for why in r["reasons"]:
            reasons[why] = reasons.get(why, 0) + 1

    summary = {
        "dataset": datasets, "run": args.run, "zero_shot": True,
        "n_recordings": len(set(r["recording"] for r in records)),
        "n_participants": len(by_part), "n_segments": int(len(records)),
        "segment_frames": args.segment_frames, "segment_seconds": args.segment_seconds,
        "mae": float(err[valid].mean()), "rmse": float(np.sqrt((err[valid] ** 2).mean())),
        "median_abs_err": float(np.median(err[valid])),
        "within_5bpm": float((err[valid] <= config.HR_TOLERANCE_BPM).mean()),
        "accepted_fraction": float(ok.mean()),
        "mae_accepted": float(acc.mean()) if len(acc) else float("nan"),
        "rmse_accepted": float(np.sqrt((acc ** 2).mean())) if len(acc) else float("nan"),
        "rejection_reasons": reasons,
        "participant_level_mae": ci,
        "per_task_mae": {k: float(np.mean(v)) for k, v in by_task.items()},
        "per_task_n": {k: len(v) for k, v in by_task.items()},
        "r_zero_lag_mean": float(np.nanmean([r["r_zero_lag"] for r in records])),
        "r_best_lag_mean": float(np.nanmean([r["r_best_lag"] for r in records])),
        "median_best_lag_s": float(np.nanmedian([r["best_lag_s"] for r in records])),
        "skipped": skipped,
        "gate": GATE,
    }

    print(f"\n=== zero-shot: {', '.join(datasets)} ===")
    print(f"  segments {summary['n_segments']}  recordings {summary['n_recordings']}  "
          f"participants {summary['n_participants']}")
    print(f"  HR MAE            {summary['mae']:.2f} bpm")
    print(f"  HR RMSE           {summary['rmse']:.2f} bpm")
    print(f"  median abs err    {summary['median_abs_err']:.2f} bpm")
    print(f"  within 5 bpm      {summary['within_5bpm']:.1%}")
    print(f"  accepted          {summary['accepted_fraction']:.1%}")
    print(f"  MAE | accepted    {summary['mae_accepted']:.2f} bpm")
    print(f"  RMSE | accepted   {summary['rmse_accepted']:.2f} bpm")
    print(f"  rejection reasons {reasons}")
    print(f"  participant MAE   mean {ci['mean']:.2f}  median {ci.get('median_subject_mae', float('nan')):.2f}  "
          f"95% CI [{ci['ci95'][0]:.2f}, {ci['ci95'][1]:.2f}] over {ci['n_subjects']} participants")
    print(f"  waveform r        {summary['r_zero_lag_mean']:.3f} at zero lag, "
          f"{summary['r_best_lag_mean']:.3f} at best lag "
          f"(median {summary['median_best_lag_s']:+.2f} s) - diagnostic only")
    if summary["per_task_mae"]:
        print("  per task:")
        for k, v in sorted(summary["per_task_mae"].items()):
            print(f"    {k:<12} MAE {v:.2f} bpm  (n={summary['per_task_n'][k]})")
    if skipped:
        print(f"  skipped/flagged recordings: {skipped}")

    out_dir = args.out or os.path.join(config.RESULTS_DIR, f"crossdataset_{args.label}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_participant": part_mae}, f, indent=2)
    cols = ["recording", "participant", "task", "fps", "end_s", "hr_ref", "hr_pred", "err",
            "snr", "margin", "cross", "r_zero_lag", "r_best_lag", "best_lag_s", "accepted"]
    with open(os.path.join(out_dir, "segments.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in sorted(records, key=lambda r: -r["err"] if np.isfinite(r["err"]) else 0):
            w.writerow([f"{r[c]:.4g}" if isinstance(r[c], float) else r[c] for c in cols])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    lo = min(np.nanmin([r["hr_ref"] for r in records]), np.nanmin([r["hr_pred"] for r in records])) - 5
    hi = max(np.nanmax([r["hr_ref"] for r in records]), np.nanmax([r["hr_pred"] for r in records])) + 5
    for acc_flag, marker, label in ((True, "o", "accepted"), (False, "x", "rejected")):
        xs = [r["hr_ref"] for r in records if r["accepted"] == acc_flag]
        ys = [r["hr_pred"] for r in records if r["accepted"] == acc_flag]
        if xs:
            ax[0].scatter(xs, ys, s=20, marker=marker, alpha=0.7, label=label)
    ax[0].plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax[0].set_xlabel("reference HR (bpm)")
    ax[0].set_ylabel("predicted HR (bpm)")
    ax[0].set_title(f"{', '.join(datasets)} zero-shot (MAE {summary['mae']:.2f} bpm)")
    ax[0].grid(alpha=0.35)
    ax[0].legend(fontsize=8)
    ax[1].hist(err[valid], bins=30)
    ax[1].set_xlabel("absolute HR error (bpm)")
    ax[1].set_ylabel("segments")
    ax[1].set_title("error distribution")
    ax[1].grid(alpha=0.35)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "zero_shot.png"), dpi=130)
    plt.close(fig)
    print(f"\nWrote {os.path.relpath(out_dir, config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

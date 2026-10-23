"""
Find and explain the worst heart-rate errors, instead of trusting average metrics.

    python -m src.failure_analysis --run results/v3_pearson --split val --top 8

Produces, in results/<run>_<split>_failures/:
    failures.csv        every segment with its error and all quality indicators
    summary.json        per-subject metrics, subject-level bootstrap CIs, error categories
    case_<n>_*.png      one diagnostic figure per selected failure

Each figure shows the original video frame, the face crop the model actually saw, the
reference and predicted waveforms, both spectra with the candidate peaks marked, and the
timing and quality numbers - enough to tell apart:

    poor video input / wrong temporal preprocessing / model error /
    HR-extraction error / incorrect reference

No segment is ever dropped to flatter the metrics. Segments excluded for a stated reason
(a reference the estimator cannot read at all) are counted and reported.
"""

import argparse
import csv
import json
import os
from typing import Dict, List, Optional

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import config
from .hr_estimators import estimate_hr_detail, peak_margin_db, periodogram
from .inference import predict_continuous, segment_signals
from .metrics import bandpass_filter, pearson_per_window, snr_db
from .models import build_model, load_weights
from .preprocessing import load_subject

DOUBLE_TOL = 0.12


def categorise(pred: float, ref: float, err: float, snr: float, stability: float) -> str:
    if not np.isfinite(pred):
        return "invalid"
    if abs(pred - 2 * ref) <= DOUBLE_TOL * 2 * ref:
        return "doubled"
    if abs(pred - 0.5 * ref) <= DOUBLE_TOL * 0.5 * ref:
        return "halved"
    if err <= config.HR_TOLERANCE_BPM:
        return "ok"
    if np.isfinite(stability) and stability > config.HR_TOLERANCE_BPM:
        return "unstable"
    if np.isfinite(snr) and snr > 3.0:
        return "wrong_despite_good_signal"
    return "too_high" if pred > ref else "too_low"


def bootstrap_subject_ci(per_subject_mae: Dict[str, float], n_boot: int = 2000,
                         seed: int = config.RANDOM_SEED) -> Dict:
    """
    Resample SUBJECTS, not windows. Overlapping windows from one subject are not independent,
    so a window-level CI would be far too narrow.
    """
    names = sorted(per_subject_mae)
    values = np.array([per_subject_mae[k] for k in names], dtype=float)
    if len(values) < 2:
        return {"mean": float(values.mean()) if len(values) else float("nan"),
                "ci95": [float("nan"), float("nan")], "n_subjects": len(values)}
    rng = np.random.default_rng(seed)
    boots = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n_boot)]
    return {
        "mean": float(values.mean()),
        "median_subject_mae": float(np.median(values)),
        "worst_subject": names[int(np.argmax(values))],
        "worst_subject_mae": float(values.max()),
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "n_subjects": len(values),
    }


def original_frame(subject: str, frame_idx: int) -> Optional[np.ndarray]:
    path = os.path.join(config.DATASET_ROOT, subject, "vid.avi")
    if not os.path.exists(path):
        return None
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None


def plot_case(case: Dict, pred: np.ndarray, target: np.ndarray, path: str) -> None:
    fps = case["fps"]
    t = np.arange(len(pred)) / fps
    pf, tf = bandpass_filter(pred, fps), bandpass_filter(target, fps)

    fig = plt.figure(figsize=(13.5, 7.6))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.15, 1.0, 1.0], hspace=0.52, wspace=0.26)

    ax = fig.add_subplot(gs[0, 0])
    frame = original_frame(case["subject"], case["start_frame"])
    if frame is not None:
        ax.imshow(frame)
    ax.set_title(f"video frame {case['start_frame']}", fontsize=9)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    if case.get("crop") is not None:
        ax.imshow(case["crop"][:, :, ::-1])  # stored BGR
    ax.set_title("face crop the model saw (64x64)", fontsize=9)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    lines = [
        f"subject        {case['subject']}  ({case['split']})",
        f"window         {case['t_start']:.1f} - {case['t_end']:.1f} s",
        f"category       {case['category'].upper()}",
        "",
        f"reference HR   {case['hr_ref']:.1f} bpm",
        f"predicted HR   {case['hr_pred']:.1f} bpm",
        f"absolute error {case['hr_err']:.1f} bpm",
        "",
        f"waveform r     {case['pearson']:.2f}",
        f"SNR            {case['snr_db']:+.1f} dB",
        f"peak margin    {case['peak_margin_db']:.1f} dB",
        f"cross-check    {case['hr_cross']:.1f} bpm (autocorr)",
        f"stability      {case['stability_bpm']:.1f} bpm",
        "",
        f"frame rate     {fps:.2f} fps",
        f"largest gap    {case['max_gap_ms']:.0f} ms",
        f"irregular dt   {case['irregular_intervals']}",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=8.4)

    ax = fig.add_subplot(gs[1, :])
    ax.plot(t, (tf - tf.mean()) / (tf.std() + 1e-9), label="reference BVP", linewidth=1.3)
    ax.plot(t, (pf - pf.mean()) / (pf.std() + 1e-9), "--", label="predicted BVP", linewidth=1.3)
    ax.set_xlabel("seconds")
    ax.set_ylabel("normalised")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("band-passed waveforms (0.7-4 Hz)", fontsize=9)

    ax = fig.add_subplot(gs[2, :])
    for sig, label, style in ((tf, "reference", "-"), (pf, "prediction", "--")):
        f, p = periodogram(sig, fps)
        m = (f >= 0.5) & (f <= 4.5)
        ax.plot(f[m] * 60, p[m] / p[m].max(), style, linewidth=1.2, label=label)
    ax.axvline(case["hr_ref"], color="tab:blue", alpha=0.5, linewidth=1)
    ax.axvline(case["hr_pred"], color="tab:orange", alpha=0.5, linewidth=1)
    ax.axvline(2 * case["hr_ref"], color="grey", linestyle=":", linewidth=1, label="2x reference")
    ax.axvline(0.5 * case["hr_ref"], color="grey", linestyle="-.", linewidth=1, label="0.5x reference")
    ax.set_xlabel("bpm")
    ax.set_ylabel("relative power")
    ax.set_xlim(30, 270)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)
    ax.set_title("spectra, with the reference rate and its harmonics marked", fontsize=9)

    fig.suptitle(
        f"{case['subject']} @ {case['t_start']:.0f}s - {case['category']} "
        f"(error {case['hr_err']:.1f} bpm)", fontsize=11,
    )
    fig.savefig(path, dpi=115, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Failure analysis for rPPG heart-rate estimates")
    p.add_argument("--run", default="results/v3_pearson")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--stride", type=int, default=config.CONTINUOUS_STRIDE)
    p.add_argument("--segment-frames", type=int, default=config.SEGMENT_FRAMES)
    p.add_argument("--method", default="fft")
    p.add_argument("--top", type=int, default=8, help="worst cases to plot")
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

    subjects = config.SPLITS[args.split]
    signals = predict_continuous(model, subjects, args.stride, cfg["normalize"], device)
    preds, targets, fps_arr, subj, hr_ref = segment_signals(signals, args.segment_frames)
    print(f"{args.run} / {args.split}: {len(preds)} segments from {len(subjects)} subjects")

    stores = {s: load_subject(s) for s in subjects}
    seg = args.segment_frames
    counters: Dict[str, int] = {}
    cases: List[Dict] = []
    excluded = 0
    seg_index: Dict[str, int] = {s: 0 for s in subjects}

    for i, (pr, tg, fps, s) in enumerate(zip(preds, targets, fps_arr, subj)):
        k = seg_index[s] * seg
        seg_index[s] += 1
        ts = stores[s]["ts"]
        pf, tf = bandpass_filter(pr, fps), bandpass_filter(tg, fps)
        d_pred = estimate_hr_detail(pf, fps, method=args.method)
        d_ref = estimate_hr_detail(tf, fps, method=args.method)
        if not np.isfinite(d_ref["hr_bpm"]):
            excluded += 1  # documented, not silently dropped
            continue
        hr_p, hr_r = d_pred["hr_bpm"], d_ref["hr_bpm"]
        err = abs(hr_p - hr_r) if np.isfinite(hr_p) else float("nan")

        # stability: spread of HR over the two halves of the segment
        halves = [estimate_hr_detail(pf[: len(pf) // 2], fps, method=args.method)["hr_bpm"],
                  estimate_hr_detail(pf[len(pf) // 2 :], fps, method=args.method)["hr_bpm"]]
        stability = abs(halves[0] - halves[1]) if all(np.isfinite(halves)) else float("nan")

        dt = np.diff(ts[k : k + seg]) if k + seg <= len(ts) else np.array([1.0 / fps])
        snr = snr_db(pf, hr_r, fps)
        case = {
            "subject": s,
            "split": args.split,
            "start_frame": int(k),
            "t_start": float(ts[k]) if k < len(ts) else k / fps,
            "t_end": float(ts[min(k + seg - 1, len(ts) - 1)]),
            "fps": float(fps),
            "hr_pred": float(hr_p), "hr_ref": float(hr_r), "hr_err": float(err),
            "hr_cross": float(estimate_hr_detail(pf, fps, method="autocorr")["hr_bpm"]),
            "hr_oximeter": float(hr_ref[i]),
            "pearson": float(pearson_per_window(pf[None], tf[None])[0]),
            "snr_db": float(snr),
            "peak_margin_db": float(peak_margin_db(d_pred)),
            "stability_bpm": float(stability),
            "max_gap_ms": float(dt.max() * 1000),
            "irregular_intervals": int((dt > 2.0 / fps).sum()),
            "pred_peaks": d_pred["peaks"],
            "ref_peaks": d_ref["peaks"],
        }
        case["category"] = categorise(hr_p, hr_r, err, snr, stability)
        counters[case["category"]] = counters.get(case["category"], 0) + 1
        case["_pred"], case["_target"] = pr, tg
        cases.append(case)

    # ---------------------------------------------------------------- summary
    per_subject: Dict[str, Dict] = {}
    for s in subjects:
        e = np.array([c["hr_err"] for c in cases if c["subject"] == s], dtype=float)
        e = e[np.isfinite(e)]
        if len(e):
            per_subject[s] = {"n_segments": int(len(e)), "mae": float(e.mean()),
                              "rmse": float(np.sqrt((e ** 2).mean())),
                              "median": float(np.median(e)), "max": float(e.max())}
    ci = bootstrap_subject_ci({k: v["mae"] for k, v in per_subject.items()})
    all_err = np.array([c["hr_err"] for c in cases if np.isfinite(c["hr_err"])])

    print(f"\n  {'subject':<10} {'n':>3} {'MAE':>7} {'RMSE':>7} {'median':>7} {'max':>7}")
    for s, m in sorted(per_subject.items(), key=lambda kv: -kv[1]["mae"]):
        print(f"  {s:<10} {m['n_segments']:>3} {m['mae']:>7.2f} {m['rmse']:>7.2f} "
              f"{m['median']:>7.2f} {m['max']:>7.2f}")
    print(f"\n  subject-level MAE  mean {ci['mean']:.2f}  median {ci['median_subject_mae']:.2f}  "
          f"95% CI [{ci['ci95'][0]:.2f}, {ci['ci95'][1]:.2f}]  (bootstrap over {ci['n_subjects']} subjects)")
    print(f"  worst subject: {ci['worst_subject']} at {ci['worst_subject_mae']:.2f} bpm")
    print(f"  window errors: p50 {np.percentile(all_err, 50):.2f}  p90 {np.percentile(all_err, 90):.2f}  "
          f"p99 {np.percentile(all_err, 99):.2f}  max {all_err.max():.2f}")
    print(f"  categories: {counters}")
    if excluded:
        print(f"  excluded {excluded} segment(s): the estimator could not read the reference "
              f"(documented, not dropped to improve metrics)")

    # ---------------------------------------------------------------- figures
    out_dir = args.out or os.path.join(config.RESULTS_DIR, f"{os.path.basename(args.run)}_{args.split}_failures")
    os.makedirs(out_dir, exist_ok=True)
    ranked = sorted([c for c in cases if np.isfinite(c["hr_err"])],
                    key=lambda c: -c["hr_err"])[: args.top]
    for n, case in enumerate(ranked, 1):
        k = case["start_frame"]
        roi = stores[case["subject"]]["roi"]
        case["crop"] = np.asarray(roi[min(k, len(roi) - 1)])
        name = f"case_{n:02d}_{case['subject']}_{case['category']}_{case['hr_err']:.1f}bpm.png"
        plot_case(case, case["_pred"], case["_target"], os.path.join(out_dir, name))
        print(f"  wrote {name}")

    with open(os.path.join(out_dir, "failures.csv"), "w", newline="", encoding="utf-8") as f:
        cols = ["subject", "start_frame", "t_start", "t_end", "fps", "category", "hr_ref",
                "hr_pred", "hr_err", "hr_cross", "hr_oximeter", "pearson", "snr_db",
                "peak_margin_db", "stability_bpm", "max_gap_ms", "irregular_intervals"]
        w = csv.writer(f)
        w.writerow(cols)
        for c in sorted(cases, key=lambda c: -c["hr_err"] if np.isfinite(c["hr_err"]) else 0):
            w.writerow([f"{c[k]:.4g}" if isinstance(c[k], float) else c[k] for k in cols])

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "run": args.run, "split": args.split, "method": args.method,
            "segment_frames": seg, "stride": args.stride,
            "n_segments": len(cases), "excluded": excluded,
            "categories": counters, "per_subject": per_subject,
            "subject_level_mae": ci,
            "window_error_percentiles": {
                "p50": float(np.percentile(all_err, 50)), "p90": float(np.percentile(all_err, 90)),
                "p99": float(np.percentile(all_err, 99)), "max": float(all_err.max()),
            },
        }, f, indent=2)
    print(f"\nWrote {os.path.relpath(out_dir, config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

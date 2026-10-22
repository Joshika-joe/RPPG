"""
Compare heart-rate estimators on a split, with the network held fixed.

Selection must happen on validation data; the test split is reserved for the final number.

    python -m src.compare_estimators --run results/v3_pearson --split val
    python -m src.compare_estimators --run results/v3_pearson --split val --bands

Reports, per estimator: MAE, RMSE, median error, % within 5 bpm, frequency-doubling and
frequency-halving counts, and how often it returns no estimate at all. The reference is the
same estimator applied to the ground-truth waveform, so the comparison isolates the
estimator's behaviour on the prediction rather than mixing in reference-side differences;
the FFT reference used elsewhere in the project is reported alongside for continuity.
"""

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch

from . import config
from .hr_estimators import METHODS, estimate_hr_detail, peak_margin_db
from .inference import predict_continuous, segment_signals
from .metrics import bandpass_filter
from .models import build_model, load_weights

DOUBLE_TOL = 0.12  # within 12% of 2x or 0.5x the reference counts as a harmonic error


def classify(pred: float, ref: float) -> str:
    if not np.isfinite(pred):
        return "invalid"
    if abs(pred - ref) <= config.HR_TOLERANCE_BPM:
        return "ok"
    if abs(pred - 2 * ref) <= DOUBLE_TOL * 2 * ref:
        return "doubled"
    if abs(pred - 0.5 * ref) <= DOUBLE_TOL * 0.5 * ref:
        return "halved"
    return "other"


def evaluate_methods(preds, targets, fps, subjects, band=config.HR_BAND_HZ,
                     subharmonic_ratio=config.HR_SUBHARMONIC_RATIO) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for method in METHODS:
        kw = {"band": band}
        if method in ("fft", "welch"):
            kw["subharmonic_ratio"] = subharmonic_ratio
        errs, classes, margins = [], [], []
        for p, t, f in zip(preds, targets, fps):
            pf, tf = bandpass_filter(p, f, band=band), bandpass_filter(t, f, band=band)
            dp = estimate_hr_detail(pf, f, method=method, **kw)
            dt = estimate_hr_detail(tf, f, method=method, **kw)
            ref = dt["hr_bpm"]
            if not np.isfinite(ref):  # reference-side failure: not the estimator's fault
                continue
            errs.append(abs(dp["hr_bpm"] - ref) if np.isfinite(dp["hr_bpm"]) else np.nan)
            classes.append(classify(dp["hr_bpm"], ref))
            margins.append(peak_margin_db(dp))
        e = np.array(errs, dtype=float)
        valid = np.isfinite(e)
        out[method] = {
            "n": int(len(e)),
            "mae": float(np.mean(e[valid])) if valid.any() else float("nan"),
            "rmse": float(np.sqrt(np.mean(e[valid] ** 2))) if valid.any() else float("nan"),
            "median": float(np.median(e[valid])) if valid.any() else float("nan"),
            "within5": float(np.mean(e[valid] <= config.HR_TOLERANCE_BPM)) if valid.any() else float("nan"),
            "doubled": int(sum(c == "doubled" for c in classes)),
            "halved": int(sum(c == "halved" for c in classes)),
            "other": int(sum(c == "other" for c in classes)),
            "invalid": int(sum(c == "invalid" for c in classes)),
            "median_peak_margin_db": float(np.nanmedian([m for m in margins if np.isfinite(m)]))
            if margins else float("nan"),
        }
    return out


def print_table(title: str, results: Dict[str, Dict]) -> None:
    print(f"\n=== {title} ===")
    print(f"  {'method':<9} {'n':>4} {'MAE':>7} {'RMSE':>7} {'med':>6} {'<=5bpm':>7} "
          f"{'x2':>3} {'x0.5':>5} {'other':>6} {'none':>5} {'margin':>7}")
    for m, r in results.items():
        print(f"  {m:<9} {r['n']:>4} {r['mae']:>7.2f} {r['rmse']:>7.2f} {r['median']:>6.2f} "
              f"{r['within5']:>6.1%} {r['doubled']:>3} {r['halved']:>5} {r['other']:>6} "
              f"{r['invalid']:>5} {r['median_peak_margin_db']:>6.1f}dB")


def main() -> None:
    p = argparse.ArgumentParser(description="Compare HR estimators (network fixed)")
    p.add_argument("--run", default="results/v3_pearson")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--stride", type=int, default=config.CONTINUOUS_STRIDE)
    p.add_argument("--segment-frames", type=int, default=config.SEGMENT_FRAMES)
    p.add_argument("--bands", action="store_true", help="also sweep the HR search band")
    p.add_argument("--harmonic", action="store_true", help="also test with the sub-harmonic fix off")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.split == "test":
        print("WARNING: estimator selection belongs on validation data. Reporting test only.\n")

    torch.set_num_threads(args.threads)
    device = config.get_device()
    run_dir = os.path.join(config.PROJECT_ROOT, args.run)
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    model = build_model(cfg["model"])
    load_weights(model, os.path.join(run_dir, "best.pth"), device)

    subjects = config.SPLITS[args.split]
    print(f"{args.run} on {args.split}: {len(subjects)} subjects, stride {args.stride}, "
          f"{args.segment_frames}-frame segments")
    signals = predict_continuous(model, subjects, args.stride, cfg["normalize"], device)
    preds, targets, fps, subj, _ = segment_signals(signals, args.segment_frames)
    print(f"{len(preds)} segments")

    report = {"run": args.run, "split": args.split, "segment_frames": args.segment_frames}

    base = evaluate_methods(preds, targets, fps, subj)
    print_table(f"estimators (band {config.HR_BAND_HZ[0]}-{config.HR_BAND_HZ[1]} Hz)", base)
    report["estimators"] = base

    if args.harmonic:
        off = evaluate_methods(preds, targets, fps, subj, subharmonic_ratio=0.0)
        print_table("sub-harmonic correction OFF", off)
        report["subharmonic_off"] = off

    if args.bands:
        report["bands"] = {}
        for band in [(0.7, 4.0), (0.7, 3.0), (0.75, 2.5), (0.8, 2.2)]:
            res = evaluate_methods(preds, targets, fps, subj, band=band)
            print_table(f"band {band[0]}-{band[1]} Hz ({band[0] * 60:.0f}-{band[1] * 60:.0f} bpm)", res)
            report["bands"][f"{band[0]}-{band[1]}"] = res

    out = args.out or os.path.join(config.RESULTS_DIR, f"estimator_comparison_{args.split}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {os.path.relpath(out, config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

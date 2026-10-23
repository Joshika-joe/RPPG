"""
Phase 7D: compare heart-rate estimation windows of 5, 8, 10 and 15 seconds.

The network is NOT retrained and its input protocol is NOT changed. v3 is fully
convolutional in time and will accept other lengths, but it was trained on 150-frame
(5 s) clips and its temporal receptive field is 125 frames, so the model input stays at
150 frames throughout. Only the duration over which heart rate is read off the
reconstructed continuous waveform varies.

Two evaluation protocols are reported:

  sliding  (primary) Every duration is evaluated at the SAME end-times, one per second,
           starting once the longest duration has full coverage. Identical timestamps
           across durations, so the comparison is not confounded by which intervals each
           duration happens to cover. Also yields the HR-over-time series used for
           stability and tracking.
  blocks   (legacy)  Non-overlapping segments, which is the existing convention; at 10 s
           it reproduces the published baseline.

Held constant: checkpoint, preprocessing, face crop, model input duration, stride,
0.7-4 Hz band, FFT estimator with its sub-harmonic rule, band-pass, overlap-add, the
reference convention, and the quality-gate thresholds.

    python -m src.compare_windows --run results/v3_pearson --split val
    python -m src.compare_windows --run results/v3_pearson --split val --recalibrate
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import config
from .hr_estimators import estimate_hr_detail, peak_margin_db
from .inference import predict_continuous
from .metrics import bandpass_filter, pearson_per_window, snr_db
from .models import build_model, load_weights

DURATIONS = (5.0, 8.0, 10.0, 15.0)
HOP_SECONDS = 1.0
REF_PROBE_SECONDS = 5.0  # fixed-length probe for measuring reference non-stationarity

# Quality-gate thresholds, calibrated on validation for 10-second windows (see src/live.py).
GATE = {"min_snr_db": -1.9, "min_peak_margin_db": 3.2, "max_disagreement_bpm": 5.7}


# ==============================================================================
# CORE MEASUREMENT
# ==============================================================================
def hr_at(sig: np.ndarray, fps: float, end_idx: int, n: int, method: str = "fft") -> Dict:
    """HR over sig[end_idx-n:end_idx], plus the quality indicators for that window."""
    seg = sig[end_idx - n : end_idx]
    if len(seg) < n or seg.std() == 0:
        return {"hr": float("nan"), "snr": float("nan"), "margin": float("nan"),
                "cross": float("nan")}
    filt = bandpass_filter(seg, fps)
    d = estimate_hr_detail(filt, fps, method=method)
    hr = d["hr_bpm"]
    return {
        "hr": float(hr),
        "snr": float(snr_db(filt, hr, fps)) if np.isfinite(hr) else float("nan"),
        "margin": float(peak_margin_db(d)),
        "cross": float(estimate_hr_detail(filt, fps, method="autocorr")["hr_bpm"]),
        "filt": filt,
    }


def gate_decision(ind: Dict, gate: Dict = GATE) -> List[str]:
    """Returns the list of failed checks; empty means accepted."""
    reasons = []
    if not np.isfinite(ind["hr"]):
        reasons.append("invalid")
        return reasons
    if np.isfinite(ind["snr"]) and ind["snr"] < gate["min_snr_db"]:
        reasons.append("snr")
    if np.isfinite(ind["margin"]) and ind["margin"] < gate["min_peak_margin_db"]:
        reasons.append("margin")
    if np.isfinite(ind["cross"]) and abs(ind["hr"] - ind["cross"]) > gate["max_disagreement_bpm"]:
        reasons.append("disagreement")
    return reasons


def reference_hr_series(gt: np.ndarray, fps: float) -> Dict[int, float]:
    """
    Reference HR from a FIXED 5-second probe at 1-second hops, independent of the
    estimation window under test. Used to quantify how much the true rate moves inside a
    window, so non-stationarity is measured the same way for every duration.
    """
    n = int(REF_PROBE_SECONDS * fps)
    hop = int(HOP_SECONDS * fps)
    out = {}
    for end in range(n, len(gt) + 1, hop):
        seg = bandpass_filter(gt[end - n : end], fps)
        out[end] = estimate_hr_detail(seg, fps, method="fft")["hr_bpm"]
    return out


def nonstationarity(ref_series: Dict[int, float], end_idx: int, n: int, fps: float) -> float:
    """Spread (max-min) of the fixed 5-s reference probe inside the window, in bpm."""
    vals = [v for k, v in ref_series.items()
            if end_idx - n <= k <= end_idx and np.isfinite(v)]
    return float(max(vals) - min(vals)) if len(vals) >= 2 else float("nan")


def evaluate_duration(signals: Dict, duration: float, protocol: str, method: str = "fft") -> Dict:
    """Evaluates one estimation duration over all subjects. Returns records + timing."""
    records: List[Dict] = []
    t_compute = 0.0
    n_calls = 0

    for subject, sig in signals.items():
        fps = sig["fps"]
        pred, gt = sig["pred"], sig["gt"]
        n = int(round(duration * fps))
        hop = int(round(HOP_SECONDS * fps))
        ref_series = sig.setdefault("_ref_series", reference_hr_series(gt, fps))

        if protocol == "sliding":
            # Common start so every duration is evaluated at identical end-times.
            first_end = int(round(max(DURATIONS) * fps))
            ends = range(first_end, len(pred) + 1, hop)
        else:
            ends = range(n, len(pred) + 1, n)

        for end in ends:
            if end - n < 0:
                continue
            t0 = time.perf_counter()
            p = hr_at(pred, fps, end, n, method)
            t_compute += time.perf_counter() - t0
            n_calls += 1
            r = hr_at(gt, fps, end, n, method)
            if not np.isfinite(r["hr"]):
                continue
            err = abs(p["hr"] - r["hr"]) if np.isfinite(p["hr"]) else float("nan")
            reasons = gate_decision(p)
            records.append({
                "subject": subject,
                "end_s": float(end / fps),
                "hr_pred": p["hr"], "hr_ref": r["hr"], "err": err,
                "snr": p["snr"], "margin": p["margin"], "cross": p["cross"],
                "pearson": float(pearson_per_window(p["filt"][None], r["filt"][None])[0])
                if np.isfinite(p["hr"]) else float("nan"),
                "ref_spread": nonstationarity(ref_series, end, n, fps),
                "accepted": not reasons,
                "reasons": reasons,
            })

    return {"records": records,
            "seconds_per_estimate": t_compute / max(n_calls, 1)}


# ==============================================================================
# SUMMARY
# ==============================================================================
def summarise(records: Sequence[Dict], duration: float, seconds_per_estimate: float) -> Dict:
    err = np.array([r["err"] for r in records], dtype=float)
    ok = np.array([r["accepted"] for r in records], dtype=bool)
    valid = np.isfinite(err)
    acc_err = err[ok & valid]
    rej_err = err[(~ok) & valid]

    # stability: change between consecutive estimates within a subject
    deltas = []
    by_subject: Dict[str, List[Dict]] = {}
    for r in records:
        by_subject.setdefault(r["subject"], []).append(r)
    for rs in by_subject.values():
        hrs = [r["hr_pred"] for r in sorted(rs, key=lambda r: r["end_s"])]
        d = np.abs(np.diff([h for h in hrs if np.isfinite(h)]))
        deltas.extend(d.tolist())

    per_subject = {}
    for s, rs in by_subject.items():
        e = np.array([r["err"] for r in rs], dtype=float)
        e = e[np.isfinite(e)]
        if len(e):
            per_subject[s] = {"n": int(len(e)), "mae": float(e.mean()),
                              "rmse": float(np.sqrt((e ** 2).mean()))}

    reasons: Dict[str, int] = {}
    for r in records:
        for why in r["reasons"]:
            reasons[why] = reasons.get(why, 0) + 1

    return {
        "duration_s": duration,
        "n_estimates": int(len(records)),
        "n_subjects": len(by_subject),
        "mae": float(np.mean(err[valid])) if valid.any() else float("nan"),
        "rmse": float(np.sqrt(np.mean(err[valid] ** 2))) if valid.any() else float("nan"),
        "median_abs_err": float(np.median(err[valid])) if valid.any() else float("nan"),
        "within_5bpm": float(np.mean(err[valid] <= config.HR_TOLERANCE_BPM)) if valid.any() else float("nan"),
        "pearson_mean": float(np.nanmean([r["pearson"] for r in records])),
        "accepted_fraction": float(ok.mean()) if len(ok) else float("nan"),
        "mae_accepted": float(acc_err.mean()) if len(acc_err) else float("nan"),
        "rmse_accepted": float(np.sqrt((acc_err ** 2).mean())) if len(acc_err) else float("nan"),
        "mae_rejected": float(rej_err.mean()) if len(rej_err) else float("nan"),
        "max_err_rejected": float(rej_err.max()) if len(rej_err) else float("nan"),
        "rejection_reasons": reasons,
        "stability_bpm": float(np.mean(deltas)) if deltas else float("nan"),
        "stability_p95_bpm": float(np.percentile(deltas, 95)) if deltas else float("nan"),
        "first_estimate_latency_s": duration,
        "update_interval_s": HOP_SECONDS,
        "freq_resolution_bpm": 60.0 / duration,
        "seconds_per_estimate": seconds_per_estimate,
        "mean_ref_spread_bpm": float(np.nanmean([r["ref_spread"] for r in records])),
        "per_subject": per_subject,
        "subject_level_mae": float(np.mean([m["mae"] for m in per_subject.values()]))
        if per_subject else float("nan"),
        "worst_subject_mae": float(max(m["mae"] for m in per_subject.values()))
        if per_subject else float("nan"),
    }


def print_table(title: str, rows: List[Dict]) -> None:
    print(f"\n=== {title} ===")
    print(f"  {'window':>7} {'n':>5} {'MAE':>7} {'RMSE':>7} {'median':>7} {'<=5bpm':>7} "
          f"{'accept':>7} {'MAE|acc':>8} {'stab':>6} {'res':>6} {'latency':>8}")
    for r in rows:
        print(f"  {r['duration_s']:>6.0f}s {r['n_estimates']:>5} {r['mae']:>7.2f} {r['rmse']:>7.2f} "
              f"{r['median_abs_err']:>7.2f} {r['within_5bpm']:>6.1%} {r['accepted_fraction']:>6.1%} "
              f"{r['mae_accepted']:>8.2f} {r['stability_bpm']:>6.2f} "
              f"{r['freq_resolution_bpm']:>5.1f}  {r['first_estimate_latency_s']:>6.0f}s")


# ==============================================================================
# PLOTS
# ==============================================================================
def plot_summary(rows: List[Dict], out_dir: str) -> None:
    d = [r["duration_s"] for r in rows]

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
    ax[0].plot(d, [r["mae"] for r in rows], "o-", label="MAE")
    ax[0].plot(d, [r["rmse"] for r in rows], "s--", label="RMSE")
    ax[0].plot(d, [r["mae_accepted"] for r in rows], "^:", label="MAE (accepted only)")
    ax[0].set_xlabel("estimation window (s)")
    ax[0].set_ylabel("bpm")
    ax[0].set_title("Error vs window duration")
    ax[0].grid(alpha=0.35)
    ax[0].legend(fontsize=8)

    ax[1].plot([r["accepted_fraction"] * 100 for r in rows], [r["mae_accepted"] for r in rows], "o-")
    for r in rows:
        ax[1].annotate(f"{r['duration_s']:.0f}s",
                       (r["accepted_fraction"] * 100, r["mae_accepted"]),
                       textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax[1].set_xlabel("accepted estimates (%)")
    ax[1].set_ylabel("MAE on accepted (bpm)")
    ax[1].set_title("Accuracy vs availability")
    ax[1].grid(alpha=0.35)

    ax[2].plot(d, [r["stability_bpm"] for r in rows], "o-", label="mean |consecutive change|")
    ax[2].plot(d, [r["freq_resolution_bpm"] for r in rows], "s--", label="FFT bin width 60/W")
    ax[2].set_xlabel("estimation window (s)")
    ax[2].set_ylabel("bpm")
    ax[2].set_title("Variability and frequency resolution")
    ax[2].grid(alpha=0.35)
    ax[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "window_summary.png"), dpi=130)
    plt.close(fig)


def plot_tracking(per_duration: Dict[float, List[Dict]], subject: str, out_dir: str,
                  highlight: Optional[float] = None) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ref_done = False
    for dur in sorted(per_duration):
        rs = sorted([r for r in per_duration[dur] if r["subject"] == subject],
                    key=lambda r: r["end_s"])
        if not rs:
            continue
        t = [r["end_s"] for r in rs]
        if not ref_done:
            ax.plot(t, [r["hr_ref"] for r in rs], "k-", linewidth=2.0, alpha=0.75,
                    label="reference (same window as each estimate)")
            ref_done = True
        ax.plot(t, [r["hr_pred"] for r in rs], "--", linewidth=1.3, label=f"{dur:.0f}s window")
        rej = [(r["end_s"], r["hr_pred"]) for r in rs if not r["accepted"]]
        if rej:
            ax.scatter([p[0] for p in rej], [p[1] for p in rej], marker="x", s=28, alpha=0.7)
    if highlight is not None:
        ax.axvline(highlight, color="red", linestyle=":", linewidth=1.4,
                   label=f"failure case at {highlight:.0f}s")
    ax.set_xlabel("time (s, window end)")
    ax.set_ylabel("heart rate (bpm)")
    ax.set_title(f"{subject}: heart rate over time by estimation window "
                 f"(x = rejected by the quality gate)")
    ax.grid(alpha=0.35)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"tracking_{subject}.png"), dpi=130)
    plt.close(fig)


# ==============================================================================
# MAIN
# ==============================================================================
def main() -> None:
    p = argparse.ArgumentParser(description="Compare HR estimation window durations")
    p.add_argument("--run", default="results/v3_pearson")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--stride", type=int, default=config.CONTINUOUS_STRIDE)
    p.add_argument("--method", default="fft", help="primary estimator; kept fixed at fft")
    p.add_argument("--recalibrate", action="store_true",
                   help="additionally derive per-duration gate thresholds from this split")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.split == "test":
        print("NOTE: the test split has already informed earlier decisions; selection must "
              "happen on validation.\n")

    torch.set_num_threads(args.threads)
    device = config.get_device()
    run_dir = os.path.join(config.PROJECT_ROOT, args.run)
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    model = build_model(cfg["model"])
    load_weights(model, os.path.join(run_dir, "best.pth"), device)

    subjects = config.SPLITS[args.split]
    print(f"{args.run} / {args.split}: {len(subjects)} subjects")
    print(f"Model input held at {config.WINDOW_FRAMES} frames (the trained length); "
          f"only the estimation window varies.")
    t0 = time.perf_counter()
    signals = predict_continuous(model, subjects, args.stride, cfg["normalize"], device)
    infer_s = time.perf_counter() - t0
    total_s = sum(len(s["pred"]) / s["fps"] for s in signals.values())
    print(f"Continuous inference: {infer_s:.1f} s for {total_s:.0f} s of video "
          f"({infer_s / total_s:.2f}x real time on {device}, {args.threads} threads)\n")

    out_dir = args.out or os.path.join(
        config.RESULTS_DIR, f"{os.path.basename(args.run)}_{args.split}_windows")
    os.makedirs(out_dir, exist_ok=True)

    report: Dict = {"run": args.run, "split": args.split, "method": args.method,
                    "stride": args.stride, "model_input_frames": config.WINDOW_FRAMES,
                    "gate": GATE, "hop_seconds": HOP_SECONDS,
                    "continuous_inference_realtime_factor": infer_s / total_s,
                    "protocols": {}}

    per_duration_records: Dict[float, List[Dict]] = {}
    for protocol in ("sliding", "blocks"):
        rows = []
        for dur in DURATIONS:
            res = evaluate_duration(signals, dur, protocol, args.method)
            rows.append(summarise(res["records"], dur, res["seconds_per_estimate"]))
            if protocol == "sliding":
                per_duration_records[dur] = res["records"]
        print_table(f"{protocol} protocol", rows)
        report["protocols"][protocol] = rows

    print("\n  Frequency resolution: a W-second window has raw FFT bins 60/W bpm apart "
          "(12.0, 7.5, 6.0, 4.0).")
    print("  Zero-padding (8x, identical for every duration) interpolates the peak location;")
    print("  it does not add information that a shorter observation did not contain.")

    print("\n  Reference non-stationarity inside each window "
          "(spread of a fixed 5-s reference probe):")
    for r in report["protocols"]["sliding"]:
        print(f"    {r['duration_s']:>4.0f}s window: mean {r['mean_ref_spread_bpm']:.2f} bpm")

    print("\n  Per-subject MAE (sliding protocol):")
    hdr = "    " + "subject".ljust(11) + "".join(f"{d:>8.0f}s" for d in DURATIONS)
    print(hdr)
    for s in subjects:
        cells = []
        for r in report["protocols"]["sliding"]:
            m = r["per_subject"].get(s)
            cells.append(f"{m['mae']:>9.2f}" if m else f"{'-':>9}")
        print("    " + s.ljust(11) + "".join(cells))

    # ---------------------------------------------------------------- plots
    plot_summary(report["protocols"]["sliding"], out_dir)
    for s in subjects:
        plot_tracking(per_duration_records, s, out_dir,
                      highlight=30.0 if s == "subject4" else None)

    # ---------------------------------------------------------------- optional recalibration
    if args.recalibrate:
        print("\n=== gate thresholds re-derived per duration (validation only) ===")
        recal = {}
        for dur, recs in per_duration_records.items():
            snr = np.array([r["snr"] for r in recs], dtype=float)
            mar = np.array([r["margin"] for r in recs], dtype=float)
            dis = np.array([abs(r["hr_pred"] - r["cross"]) for r in recs], dtype=float)
            recal[dur] = {
                "min_snr_db": float(np.nanpercentile(snr, 5)),
                "min_peak_margin_db": float(np.nanpercentile(mar, 5)),
                "max_disagreement_bpm": float(np.nanpercentile(dis, 95)),
            }
            print(f"  {dur:>4.0f}s  SNR p5 {recal[dur]['min_snr_db']:+6.2f} dB   "
                  f"margin p5 {recal[dur]['min_peak_margin_db']:6.2f} dB   "
                  f"disagreement p95 {recal[dur]['max_disagreement_bpm']:5.2f} bpm")
        print("  (thresholds move with duration, so the fixed 10-s values do not carry over;")
        print("   reported separately from the fixed-threshold comparison above)")
        report["recalibrated_thresholds"] = {str(k): v for k, v in recal.items()}

    with open(os.path.join(out_dir, "windows.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {os.path.relpath(out_dir, config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

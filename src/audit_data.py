"""
Audit the dataset and the evaluation methodology itself (Phase Six).

    python -m src.audit_data

Checks, in order:
  A. Subject-level separation - splits disjoint, and no window can cross a split boundary.
  D. Reference-signal integrity - missing samples, implausible oximeter values, timestamp
     irregularity, clipping, frame/signal length mismatch, and reference windows whose
     spectrum is ambiguous between a rate and its double.

Nothing here modifies ground truth. Findings are reported so that any correction can be
justified separately; `results/data_audit.json` records them.
"""

import argparse
import json
import os
from typing import Dict, List

import numpy as np

from . import config
from .dataset import WindowDataset
from .hr_estimators import estimate_hr_detail
from .preprocessing import load_subject, read_ground_truth

PLAUSIBLE_HR = (35.0, 220.0)  # outside this a resting-adult reference value is suspect


# ==============================================================================
# A. SUBJECT-LEVEL SEPARATION
# ==============================================================================
def audit_splits() -> Dict:
    splits = {name: list(subjects) for name, subjects in config.SPLITS.items()}
    sets = {k: set(v) for k, v in splits.items()}
    findings: List[str] = []

    overlaps = {}
    for a in sets:
        for b in sets:
            if a < b:
                shared = sets[a] & sets[b]
                if shared:
                    overlaps[f"{a}&{b}"] = sorted(shared)
                    findings.append(f"subjects in both {a} and {b}: {sorted(shared)}")

    dupes = {k: [s for s in set(v) if v.count(s) > 1] for k, v in splits.items()}
    for k, d in dupes.items():
        if d:
            findings.append(f"duplicate entries in {k}: {d}")

    on_disk = set(os.listdir(config.SUBJECT_DIR)) if os.path.isdir(config.SUBJECT_DIR) else set()
    listed = set().union(*sets.values())
    unused = sorted(on_disk - listed)
    missing = sorted(listed - on_disk)
    if missing:
        findings.append(f"listed in a split but not preprocessed: {missing}")

    # Windows are generated per subject inside WindowDataset, so a window cannot span two
    # subjects by construction. Verify that the datasets built from the splits are disjoint
    # in the subjects they actually load.
    loaded = {}
    for name, subjects in splits.items():
        ds = WindowDataset(subjects, stride=config.EVAL_STRIDE)
        loaded[name] = set(ds.subjects)
    window_overlap = {}
    for a in loaded:
        for b in loaded:
            if a < b and (loaded[a] & loaded[b]):
                window_overlap[f"{a}&{b}"] = sorted(loaded[a] & loaded[b])
                findings.append(f"windows from the same subject in {a} and {b}")

    return {
        "sizes": {k: len(v) for k, v in splits.items()},
        "subject_overlap": overlaps,
        "window_source_overlap": window_overlap,
        "preprocessed_but_unused": unused,
        "listed_but_missing": missing,
        "findings": findings,
        "clean": not findings,
    }


# ==============================================================================
# D. REFERENCE-SIGNAL INTEGRITY
# ==============================================================================
def audit_subject_reference(subject: str, segment_frames: int = config.SEGMENT_FRAMES) -> Dict:
    store = load_subject(subject)
    bvp, hr, ts = store["bvp"], store["hr"], store["ts"]
    meta = store["meta"]
    fps = meta["fps"]
    issues: List[str] = []

    gt_path = os.path.join(config.DATASET_ROOT, subject, "ground_truth.txt")
    raw_len = None
    if os.path.exists(gt_path):
        raw_bvp, _, _ = read_ground_truth(gt_path)
        raw_len = len(raw_bvp)

    n_missing = int(np.sum(~np.isfinite(bvp)))
    if n_missing:
        issues.append(f"{n_missing} non-finite BVP samples")

    # Oximeter HR line
    hr_valid = np.isfinite(hr)
    implausible = int(np.sum((hr < PLAUSIBLE_HR[0]) | (hr > PLAUSIBLE_HR[1])))
    frac_implausible = implausible / max(len(hr), 1)
    if frac_implausible > 0.01:
        issues.append(f"oximeter HR implausible in {frac_implausible:.0%} of samples "
                      f"(min {hr[hr_valid].min():.0f}, max {hr[hr_valid].max():.0f})")

    # Timestamp regularity
    dt = np.diff(ts)
    jitter = float(np.std(dt) / np.mean(dt)) if len(dt) and np.mean(dt) > 0 else float("nan")
    max_gap = float(dt.max()) if len(dt) else float("nan")
    if np.isfinite(max_gap) and max_gap > 3.0 / fps:
        issues.append(f"largest frame interval {max_gap * 1000:.0f} ms (> 3 frames)")
    if not np.all(dt > 0):
        issues.append("non-monotonic timestamps")

    # Clipping / saturation of the reference waveform
    lo, hi = float(bvp.min()), float(bvp.max())
    at_extreme = int(np.sum((bvp <= lo + 1e-6) | (bvp >= hi - 1e-6)))
    frac_clip = at_extreme / max(len(bvp), 1)
    if frac_clip > 0.02:
        issues.append(f"{frac_clip:.1%} of BVP samples sit at the signal's min/max (clipping)")

    if meta.get("gt_resampled_to_frames"):
        issues.append("ground truth was interpolated onto frame times (length mismatch)")

    # Harmonic ambiguity in the REFERENCE waveform itself
    ambiguous = []
    for start in range(0, len(bvp) - segment_frames + 1, segment_frames):
        seg = bvp[start : start + segment_frames]
        d = estimate_hr_detail(seg, fps, method="fft")
        naive = estimate_hr_detail(seg, fps, method="fft", subharmonic_ratio=0.0)["hr_bpm"]
        if np.isfinite(d["hr_bpm"]) and abs(naive - 2 * d["hr_bpm"]) < 0.15 * 2 * d["hr_bpm"]:
            ambiguous.append({"start_frame": int(start),
                              "corrected_bpm": round(float(d["hr_bpm"]), 1),
                              "uncorrected_bpm": round(float(naive), 1)})
    if ambiguous:
        issues.append(f"{len(ambiguous)} reference segment(s) where the 2nd harmonic "
                      f"outweighs the fundamental")

    return {
        "subject": subject,
        "split": next((k for k, v in config.SPLITS.items() if subject in v), "unassigned"),
        "n_frames": int(meta["n_frames"]),
        "raw_gt_samples": raw_len,
        "fps": round(float(fps), 2),
        "timestamp_jitter": round(jitter, 4) if np.isfinite(jitter) else None,
        "max_frame_gap_ms": round(max_gap * 1000, 1) if np.isfinite(max_gap) else None,
        "oximeter_hr_range": [round(float(hr[hr_valid].min()), 1), round(float(hr[hr_valid].max()), 1)]
        if hr_valid.any() else None,
        "oximeter_implausible_frac": round(frac_implausible, 4),
        "bvp_clipped_frac": round(frac_clip, 4),
        "harmonic_ambiguous_segments": ambiguous,
        "issues": issues,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Audit splits and reference-signal integrity")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    print("=" * 78)
    print("A. SUBJECT-LEVEL SEPARATION")
    print("=" * 78)
    splits = audit_splits()
    print(f"  split sizes: {splits['sizes']}")
    print(f"  preprocessed but in no split: {splits['preprocessed_but_unused'] or 'none'}")
    if splits["clean"]:
        print("  PASS - splits are disjoint and no window source is shared between them")
    else:
        for f in splits["findings"]:
            print(f"  FAIL - {f}")

    print()
    print("=" * 78)
    print("D. REFERENCE-SIGNAL INTEGRITY")
    print("=" * 78)
    subjects = [s for s in config.ALL_SUBJECTS]
    reports = []
    print(f"  {'subject':<10} {'split':<6} {'fps':>6} {'jitter':>7} {'gap ms':>7} "
          f"{'oximeter HR':>14} {'clip':>6}  issues")
    for s in subjects:
        try:
            r = audit_subject_reference(s)
        except FileNotFoundError:
            print(f"  {s:<10} not preprocessed - skipped")
            continue
        reports.append(r)
        ox = f"{r['oximeter_hr_range'][0]:.0f}-{r['oximeter_hr_range'][1]:.0f}" if r["oximeter_hr_range"] else "-"
        print(f"  {r['subject']:<10} {r['split']:<6} {r['fps']:>6.2f} "
              f"{(r['timestamp_jitter'] or 0):>7.3f} {(r['max_frame_gap_ms'] or 0):>7.0f} "
              f"{ox:>14} {r['bvp_clipped_frac']:>6.1%}  {len(r['issues'])}")

    print("\n  Subjects with findings:")
    any_issue = False
    for r in reports:
        if r["issues"]:
            any_issue = True
            print(f"    {r['subject']} ({r['split']}):")
            for i in r["issues"]:
                print(f"      - {i}")
    if not any_issue:
        print("    none")

    fps_values = sorted({round(r["fps"], 1) for r in reports})
    print(f"\n  Distinct frame rates in the dataset: {fps_values}")
    print("  (a single assumed rate would bias HR for every subject not at that rate)")

    out = args.out or os.path.join(config.RESULTS_DIR, "data_audit.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"splits": splits, "subjects": reports}, f, indent=2)
    print(f"\nWrote {os.path.relpath(out, config.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

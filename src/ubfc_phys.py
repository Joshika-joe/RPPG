"""
Adapter for UBFC-Phys, used for zero-shot cross-dataset evaluation.

This converts UBFC-Phys recordings into the same per-subject store that
src/preprocessing.py writes for UBFC-rPPG, so the model, windowing, inference and metric
code are reused unchanged. It writes to a SEPARATE root so the UBFC-rPPG cache and results
are untouched.

Dataset facts (from the official dataset page and the dataset paper; see docs/DATASETS.md):
    56 participants, three tasks each - T1 rest, T2 speech, T3 arithmetic (168 recordings)
    video  Edmund Optics EO-23121C, 1024x1024, 35 fps, .avi
    BVP    Empatica E4 wrist band, 64 Hz, .csv (one value per line)
    layout s<N>/vid_s<N>_T<k>.avi  and  s<N>/bvp_s<N>_T<k>.csv

Differences from UBFC-rPPG that matter for interpretation:
    * 35 fps, not ~29.5. The windowing resamples onto the model's 30 Hz grid, so this is an
      input-format adaptation, not a model change.
    * 1024x1024, not 640x480, so the face occupies a different pixel scale before cropping.
    * The reference is a WRIST BVP sensor, not a finger oximeter. Wrist pulse arrives later
      than facial pulse (pulse transit time) and the E4 is not hardware-synchronised with the
      camera, so a fixed unknown lag may exist between the two signals. Heart RATE is
      unaffected by a constant lag; waveform CORRELATION is, and is reported with that caveat.
    * No per-frame timestamps are published, so frame times are derived from the container's
      frame rate. UBFC-rPPG, by contrast, ships real (irregular) timestamps.

    python -m src.ubfc_phys --root /path/to/UBFC-Phys            # preprocess all
    python -m src.ubfc_phys --root /path/to/UBFC-Phys --tasks T1 # rest task only
"""

import argparse
import csv
import json
import os
import re
import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import config
from .preprocessing import crop_resize, detect_face_box, get_face_detector, save_subject

BVP_HZ = 64.0  # Empatica E4 blood volume pulse
NOMINAL_FPS = 35.0
TASKS = ("T1", "T2", "T3")
TASK_NAMES = {"T1": "rest", "T2": "speech", "T3": "arithmetic"}

# Store root for this dataset, kept apart from the UBFC-rPPG cache.
SUBJECT_DIR = os.path.join(config.CHECKPOINT_DIR, "subjects_ubfc_phys")


def read_bvp_csv(path: str) -> np.ndarray:
    """
    One BVP value per line at 64 Hz. Tolerates a header line and a leading index column,
    both of which appear in redistributed copies.
    """
    values: List[float] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            cell = row[-1].strip()
            if not cell:
                continue
            try:
                values.append(float(cell))
            except ValueError:
                continue  # header or label
    if not values:
        raise ValueError(f"{path}: no numeric BVP samples found")
    return np.asarray(values, dtype=np.float32)


def find_recordings(root: str, tasks: Sequence[str] = TASKS,
                    participants: Optional[Sequence[str]] = None) -> List[Dict]:
    """Discovers s<N>/vid_s<N>_T<k>.avi plus its bvp csv. Returns one entry per recording."""
    found: List[Dict] = []
    if not os.path.isdir(root):
        return found
    for entry in sorted(os.listdir(root), key=lambda s: (len(s), s)):
        if not re.fullmatch(r"s\d+", entry):
            continue
        if participants and entry not in participants:
            continue
        folder = os.path.join(root, entry)
        if not os.path.isdir(folder):
            continue
        for task in tasks:
            video = os.path.join(folder, f"vid_{entry}_{task}.avi")
            bvp = os.path.join(folder, f"bvp_{entry}_{task}.csv")
            if os.path.exists(video) and os.path.exists(bvp):
                found.append({"recording": f"{entry}_{task}", "participant": entry,
                              "task": task, "video": video, "bvp": bvp})
    return found


def process_recording(rec: Dict, target_size: Tuple[int, int] = config.TARGET_SIZE,
                      n_detect_frames: int = config.FACE_DETECT_FRAMES,
                      cascade: Optional[cv2.CascadeClassifier] = None,
                      bvp_hz: float = BVP_HZ) -> Optional[Dict]:
    """
    Streams one recording and returns a store dict in the project's standard format, or None
    if it cannot be used. Face detection and cropping reuse the UBFC-rPPG code unchanged.
    """
    if cascade is None:
        cascade = get_face_detector()

    bvp_raw = read_bvp_csv(rec["bvp"])

    cap = cv2.VideoCapture(rec["video"])
    if not cap.isOpened():
        print(f"  [{rec['recording']}] cannot open video - skipped")
        return None
    container_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = container_fps if container_fps and container_fps > 1.0 else NOMINAL_FPS

    buffer: List[np.ndarray] = []
    box = None
    rois: List[np.ndarray] = []
    frame_size = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_size is None:
            frame_size = (int(frame.shape[0]), int(frame.shape[1]))
        if box is None:
            buffer.append(frame)
            if len(buffer) >= n_detect_frames:
                box = detect_face_box(buffer, cascade)
                if box is None:
                    cap.release()
                    print(f"  [{rec['recording']}] no face in first {n_detect_frames} frames - skipped")
                    return None
                rois.extend(crop_resize(f, box, target_size) for f in buffer)
                buffer = []
        else:
            rois.append(crop_resize(frame, box, target_size))
    cap.release()

    if box is None:  # shorter than the detection window
        box = detect_face_box(buffer, cascade)
        if box is None:
            print(f"  [{rec['recording']}] no face found - skipped")
            return None
        rois.extend(crop_resize(f, box, target_size) for f in buffer)

    n_frames = len(rois)
    if n_frames < config.WINDOW_FRAMES:
        print(f"  [{rec['recording']}] only {n_frames} frames - skipped")
        return None

    # No published per-frame timestamps: derive frame times from the container rate, and
    # resample the 64 Hz wrist BVP onto them. Both signals are assumed to start together;
    # see the module docstring on synchronisation.
    ts = (np.arange(n_frames) / fps).astype(np.float32)
    bvp_times = np.arange(len(bvp_raw)) / bvp_hz
    bvp = np.interp(ts, bvp_times, bvp_raw).astype(np.float32)
    coverage = float(min(ts[-1], bvp_times[-1]) / ts[-1]) if ts[-1] > 0 else 0.0

    meta = {
        "subject": rec["recording"],
        "dataset": "UBFC-Phys",
        "participant": rec["participant"],
        "task": rec["task"],
        "task_name": TASK_NAMES.get(rec["task"], rec["task"]),
        "n_frames": int(n_frames),
        "fps": float(fps),
        "fps_source": "container" if container_fps and container_fps > 1.0 else "nominal",
        "duration_s": float(ts[-1]),
        "face_box_xywh": [int(v) for v in box],
        "frame_size_hw": list(frame_size),
        "roi_size_hw": list(target_size),
        "channel_order": "BGR",
        "gt_resampled_to_frames": True,  # 64 Hz -> frame rate, always
        "reference_sensor": "Empatica E4 wrist BVP",
        "reference_hz": bvp_hz,
        "reference_coverage": coverage,
        "timestamps": "derived from container fps (dataset ships none)",
    }
    return {
        "roi": np.stack(rois).astype(np.uint8),
        "bvp": bvp,
        # No reference heart-rate channel is published; carry NaN so downstream code that
        # reads hr_ref reports "unavailable" rather than a fabricated number.
        "hr": np.full(n_frames, np.nan, dtype=np.float32),
        "ts": ts,
        "meta": meta,
    }


def run(root: str, tasks: Sequence[str] = TASKS, participants: Optional[Sequence[str]] = None,
        out_root: str = SUBJECT_DIR, force: bool = False, limit: Optional[int] = None) -> List[str]:
    recordings = find_recordings(root, tasks, participants)
    if not recordings:
        raise SystemExit(
            f"No UBFC-Phys recordings under {root!r}. Expected s<N>/vid_s<N>_T<k>.avi "
            f"alongside s<N>/bvp_s<N>_T<k>.csv - see docs/DATASETS.md."
        )
    if limit:
        recordings = recordings[:limit]
    cascade = get_face_detector()
    done: List[str] = []
    print(f"UBFC-Phys: {len(recordings)} recording(s) -> {out_root}")
    t_all = time.time()
    for i, rec in enumerate(recordings, 1):
        target = os.path.join(out_root, rec["recording"])
        if not force and os.path.exists(os.path.join(target, "meta.json")):
            print(f"  [{i:03d}/{len(recordings)}] {rec['recording']}: cached")
            done.append(rec["recording"])
            continue
        t0 = time.time()
        data = process_recording(rec, cascade=cascade)
        if data is None:
            continue
        save_subject(data, out_root)
        m = data["meta"]
        print(f"  [{i:03d}/{len(recordings)}] {rec['recording']} ({m['task_name']}): "
              f"{m['n_frames']} frames, {m['fps']:.2f} fps ({m['fps_source']}), "
              f"ref coverage {m['reference_coverage']:.2f}  [{time.time() - t0:.1f}s]")
        done.append(rec["recording"])
    print(f"Done: {len(done)}/{len(recordings)} in {time.time() - t_all:.0f}s")
    return done


def main() -> None:
    p = argparse.ArgumentParser(description="Preprocess UBFC-Phys into the project's store format")
    p.add_argument("--root", required=True, help="directory containing s1/, s2/, ...")
    p.add_argument("--tasks", nargs="*", default=list(TASKS), choices=list(TASKS))
    p.add_argument("--participants", nargs="*", default=None, help="e.g. s1 s2 s3")
    p.add_argument("--out-root", default=SUBJECT_DIR)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    run(args.root, args.tasks, args.participants, args.out_root, args.force, args.limit)


if __name__ == "__main__":
    main()

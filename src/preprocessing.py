"""
Per-subject preprocessing for the UBFC-rPPG dataset.

For each subject this decodes vid.avi ONCE, streaming frame-by-frame (never holding the
full video in RAM), estimates a single robust face box from the first N frames, crops and
resizes every frame to a 64x64 ROI, aligns BVP / HR / timestamps per frame, and writes:

    checkpoints/subjects/<subject>/
        roi.npy    uint8   (T, 64, 64, 3)   BGR (OpenCV order) face ROI per frame
        bvp.npy    float32 (T,)             raw BVP aligned to frames
        hr.npy     float32 (T,)             reference HR (bpm) from the pulse oximeter
        ts.npy     float32 (T,)             frame timestamps (s)
        meta.json                           fps, face box, flags

Windowing / normalization / augmentation are NOT done here - see dataset.WindowDataset.
That keeps stride, window length and augmentation as training-time choices instead of a
30-minute re-preprocess.

Usage:
    python -m src.preprocessing            # all subjects in splits/, skip already processed
    python -m src.preprocessing --force    # regenerate everything
    python -m src.preprocessing --subjects subject1 subject3
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import config

SUBJECT_FILES = {
    "roi": "roi.npy",
    "bvp": "bvp.npy",
    "hr": "hr.npy",
    "ts": "ts.npy",
    "meta": "meta.json",
}

Box = Tuple[int, int, int, int]  # x, y, w, h


# ==============================================================================
# RAW INPUT
# ==============================================================================
def get_face_detector() -> cv2.CascadeClassifier:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")
    return cascade


def read_ground_truth(gt_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    UBFC-rPPG ground_truth.txt format (3 lines):
        line 1: BVP waveform
        line 2: heart rate (bpm) from the pulse oximeter
        line 3: timestamps (s)
    """
    with open(gt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) < 3:
        raise ValueError(f"{gt_path}: expected 3 lines, got {len(lines)}")
    bvp = np.array(lines[0].split(), dtype=np.float32)
    hr = np.array(lines[1].split(), dtype=np.float32)
    ts = np.array(lines[2].split(), dtype=np.float32)
    if not (len(bvp) == len(hr) == len(ts)):
        raise ValueError(
            f"{gt_path}: length mismatch bvp={len(bvp)} hr={len(hr)} ts={len(ts)}"
        )
    return bvp, hr, ts


# ==============================================================================
# FACE ROI
# ==============================================================================
def detect_face_box(
    frames: Sequence[np.ndarray], cascade: cv2.CascadeClassifier
) -> Optional[Box]:
    """
    Detects the LARGEST face in each frame and returns the per-coordinate median box.
    The median suppresses single-frame Haar jitter / false positives.
    """
    boxes = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=config.FACE_SCALE_FACTOR,
            minNeighbors=config.FACE_MIN_NEIGHBORS,
            minSize=config.FACE_MIN_SIZE,
        )
        if len(faces) > 0:
            boxes.append(max(faces, key=lambda b: b[2] * b[3]))
    if not boxes:
        return None
    box = np.median(np.array(boxes, dtype=np.float32), axis=0)
    x, y, w, h = np.round(box).astype(int)
    return int(x), int(y), int(w), int(h)


def crop_resize(frame: np.ndarray, box: Box, target_size: Tuple[int, int]) -> np.ndarray:
    """Crops `box` (clipped to the frame) and resizes to target_size (H, W). Returns uint8."""
    x, y, w, h = box
    fh, fw = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + w), min(fh, y + h)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        raise ValueError(f"Empty crop for box {box} on frame {frame.shape}")
    # cv2.resize takes (W, H)
    return cv2.resize(crop, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)


# ==============================================================================
# PER-SUBJECT PIPELINE
# ==============================================================================
def process_subject(
    subject: str,
    dataset_root: str = config.DATASET_ROOT,
    target_size: Tuple[int, int] = config.TARGET_SIZE,
    n_detect_frames: int = config.FACE_DETECT_FRAMES,
    cascade: Optional[cv2.CascadeClassifier] = None,
) -> Optional[Dict]:
    """
    Streams one subject's video, returns dict(roi, bvp, hr, ts, meta) or None if unusable.
    """
    subject_path = os.path.join(dataset_root, subject)
    video_path = os.path.join(subject_path, "vid.avi")
    gt_path = os.path.join(subject_path, "ground_truth.txt")

    if not os.path.exists(video_path) or not os.path.exists(gt_path):
        print(f"  [{subject}] missing vid.avi or ground_truth.txt - skipped")
        return None

    if cascade is None:
        cascade = get_face_detector()

    bvp, hr, ts = read_ground_truth(gt_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [{subject}] cannot open video - skipped")
        return None

    frame_size = None
    buffer: List[np.ndarray] = []  # first frames, kept until the face box is known
    box: Optional[Box] = None
    rois: List[np.ndarray] = []

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
                    print(f"  [{subject}] no face in first {n_detect_frames} frames - skipped")
                    return None
                rois.extend(crop_resize(f, box, target_size) for f in buffer)
                buffer = []
        else:
            rois.append(crop_resize(frame, box, target_size))
    cap.release()

    # Short video: fewer frames than n_detect_frames
    if box is None:
        box = detect_face_box(buffer, cascade)
        if box is None:
            print(f"  [{subject}] no face found - skipped")
            return None
        rois.extend(crop_resize(f, box, target_size) for f in buffer)

    n_frames = len(rois)
    if n_frames == 0:
        print(f"  [{subject}] video has 0 frames - skipped")
        return None

    # Align ground truth to frames. UBFC-rPPG is 1:1 (one BVP sample per frame); if a
    # subject ever mismatches, interpolate onto evenly spaced frame times instead.
    resampled = False
    if n_frames != len(bvp):
        resampled = True
        frame_times = np.linspace(ts[0], ts[-1], n_frames).astype(np.float32)
        bvp = np.interp(frame_times, ts, bvp).astype(np.float32)
        hr = np.interp(frame_times, ts, hr).astype(np.float32)
        ts = frame_times

    duration = float(ts[-1] - ts[0])
    fps = (n_frames - 1) / duration if duration > 0 else float("nan")

    meta = {
        "subject": subject,
        "n_frames": int(n_frames),
        "fps": float(fps),
        "duration_s": duration,
        "face_box_xywh": [int(v) for v in box],
        "frame_size_hw": list(frame_size),
        "roi_size_hw": list(target_size),
        "channel_order": "BGR",
        "gt_resampled_to_frames": resampled,
    }
    return {
        "roi": np.stack(rois).astype(np.uint8),
        "bvp": bvp.astype(np.float32),
        "hr": hr.astype(np.float32),
        "ts": ts.astype(np.float32),
        "meta": meta,
    }


# ==============================================================================
# STORAGE
# ==============================================================================
def subject_dir(subject: str, root: str = config.SUBJECT_DIR) -> str:
    return os.path.join(root, subject)


def subject_is_processed(subject: str, root: str = config.SUBJECT_DIR) -> bool:
    d = subject_dir(subject, root)
    return all(os.path.exists(os.path.join(d, f)) for f in SUBJECT_FILES.values())


def save_subject(data: Dict, root: str = config.SUBJECT_DIR) -> str:
    d = subject_dir(data["meta"]["subject"], root)
    os.makedirs(d, exist_ok=True)
    for key in ("roi", "bvp", "hr", "ts"):
        np.save(os.path.join(d, SUBJECT_FILES[key]), data[key])
    with open(os.path.join(d, SUBJECT_FILES["meta"]), "w", encoding="utf-8") as f:
        json.dump(data["meta"], f, indent=2)
    return d


def load_subject(subject: str, root: str = config.SUBJECT_DIR, mmap: bool = True) -> Dict:
    """Loads one processed subject. With mmap=True the ROI array is memory-mapped (lazy)."""
    d = subject_dir(subject, root)
    if not subject_is_processed(subject, root):
        raise FileNotFoundError(
            f"Subject '{subject}' not preprocessed in {root}. Run: python -m src.preprocessing"
        )
    roi = np.load(os.path.join(d, SUBJECT_FILES["roi"]), mmap_mode="r" if mmap else None)
    with open(os.path.join(d, SUBJECT_FILES["meta"]), "r", encoding="utf-8") as f:
        meta = json.load(f)
    return {
        "roi": roi,
        "bvp": np.load(os.path.join(d, SUBJECT_FILES["bvp"])),
        "hr": np.load(os.path.join(d, SUBJECT_FILES["hr"])),
        "ts": np.load(os.path.join(d, SUBJECT_FILES["ts"])),
        "meta": meta,
    }


# ==============================================================================
# ENTRY POINT
# ==============================================================================
def run_preprocessing(
    subjects: Optional[Sequence[str]] = None,
    force: bool = False,
    root: str = config.SUBJECT_DIR,
) -> List[str]:
    """Processes every subject not yet on disk (or all, with force=True). Returns processed names."""
    subjects = list(subjects) if subjects else config.ALL_SUBJECTS
    cascade = get_face_detector()
    done: List[str] = []
    t_all = time.time()

    print(f"Preprocessing {len(subjects)} subjects -> {root}")
    for i, subject in enumerate(subjects, 1):
        if not force and subject_is_processed(subject, root):
            print(f"  [{i:02d}/{len(subjects)}] {subject}: cached")
            done.append(subject)
            continue
        t0 = time.time()
        data = process_subject(subject, cascade=cascade)
        if data is None:
            continue
        save_subject(data, root)
        m = data["meta"]
        print(
            f"  [{i:02d}/{len(subjects)}] {subject}: {m['n_frames']} frames, "
            f"{m['fps']:.2f} fps, box={m['face_box_xywh']}"
            f"{' (gt resampled)' if m['gt_resampled_to_frames'] else ''}"
            f"  [{time.time() - t0:.1f}s]"
        )
        done.append(subject)

    print(f"Done: {len(done)}/{len(subjects)} subjects in {time.time() - t_all:.0f}s")
    return done


def main() -> None:
    p = argparse.ArgumentParser(description="Per-subject rPPG preprocessing")
    p.add_argument("--force", action="store_true", help="regenerate even if cached")
    p.add_argument("--subjects", nargs="*", default=None, help="subset of subject names")
    args = p.parse_args()
    run_preprocessing(subjects=args.subjects, force=args.force)


if __name__ == "__main__":
    main()

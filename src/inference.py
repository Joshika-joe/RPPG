"""
Continuous (per-subject) inference by overlap-add.

Instead of scoring isolated 5-s windows, slide the model over the whole recording with a
short stride, standardize each window's prediction, and blend the overlapping predictions
with a Hann weight into one continuous BVP per subject. HR is then estimated on longer,
non-overlapping segments (default 300 frames ~ 10 s) - the way UBFC results are usually
reported - and single-window flukes average out.

    signals = predict_continuous(model, subjects, stride=30, normalize="temporal", device=dev)
    preds, targets, fps, subj, hr_ref = segment_signals(signals, segment_frames=300)
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import config
from .dataset import WindowDataset, normalize_target
from .models import run_model


def hann_weights(n: int) -> np.ndarray:
    """Hann window without the zero end-points (so every frame gets positive weight)."""
    return np.hanning(n + 2)[1:-1].astype(np.float64)


def overlap_add(
    windows: Sequence[Tuple[int, np.ndarray]], n_frames: int, window_frames: int
) -> np.ndarray:
    """
    windows: list of (start, prediction (window_frames,)). Each prediction is standardized
    before blending so amplitude differences between windows do not create seams.
    """
    out = np.zeros(n_frames, dtype=np.float64)
    wsum = np.zeros(n_frames, dtype=np.float64)
    w = hann_weights(window_frames)
    for start, pred in windows:
        p = np.asarray(pred, dtype=np.float64)
        p = (p - p.mean()) / (p.std() + config.EPSILON)
        out[start : start + window_frames] += w * p
        wsum[start : start + window_frames] += w
    covered = wsum > 0
    out[covered] /= wsum[covered]
    return out.astype(np.float32)


@torch.no_grad()
def predict_continuous(
    model: torch.nn.Module,
    subjects: Sequence[str],
    stride: int,
    normalize: str,
    device: torch.device,
    batch_size: int = config.BATCH_SIZE,
    window_frames: int = config.WINDOW_FRAMES,
) -> Dict[str, Dict]:
    """
    Returns {subject: {"pred": (T,), "gt": (T,), "hr_ref": (T,), "fps": float, "n_windows": int}}
    with pred the overlap-added continuous BVP estimate and gt the raw BVP aligned to frames.
    """
    model.eval()
    ds = WindowDataset(
        subjects, window_frames=window_frames, stride=stride, normalize=normalize,
        augment=False, cover_tail=True,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    per_subject: Dict[str, List[Tuple[int, np.ndarray]]] = {s: [] for s in ds.subjects}
    for x, _, meta in loader:
        out = run_model(model, x, meta, device).cpu().numpy()
        for pred, subj, start in zip(out, meta["subject"], meta["start"].tolist()):
            per_subject[subj].append((start, pred))

    signals: Dict[str, Dict] = {}
    for si, subj in enumerate(ds.subjects):
        store = ds.stores[si]
        n = int(store["meta"]["n_frames"])
        signals[subj] = {
            "pred": overlap_add(per_subject[subj], n, window_frames),
            "gt": np.asarray(store["bvp"], dtype=np.float32),
            "hr_ref": np.asarray(store["hr"], dtype=np.float32),
            "fps": float(store["meta"]["fps"]),
            "n_windows": len(per_subject[subj]),
        }
    return signals


def segment_signals(
    signals: Dict[str, Dict], segment_frames: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    Cuts every subject's continuous prediction and ground truth into non-overlapping
    segments of `segment_frames`, standardizing each. Returns arrays shaped for
    metrics.compute_metrics: preds (N, L), targets (N, L), fps (N,), subjects (N,), hr_ref (N,).
    """
    preds, targets, fps, subj, hr_ref = [], [], [], [], []
    for name, sig in signals.items():
        n = len(sig["pred"])
        for start in range(0, n - segment_frames + 1, segment_frames):
            end = start + segment_frames
            preds.append(normalize_target(sig["pred"][start:end]))
            targets.append(normalize_target(sig["gt"][start:end]))
            fps.append(sig["fps"])
            subj.append(name)
            hr_ref.append(float(sig["hr_ref"][start:end].mean()))
    return np.stack(preds), np.stack(targets), np.array(fps), subj, np.array(hr_ref)

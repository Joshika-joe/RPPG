"""Shared fixtures: a synthetic preprocessed subject on disk (no video / face detection)."""

import json
import os

import numpy as np
import pytest

FPS = 30.0
N_FRAMES = 450  # 15 s
HR_BPM = 72.0


def make_bvp(n: int, fps: float, hr_bpm: float, phase: float = 0.0) -> np.ndarray:
    t = np.arange(n) / fps
    f0 = hr_bpm / 60.0
    # pulse-like: fundamental + weaker 2nd harmonic
    return (np.sin(2 * np.pi * f0 * t + phase) + 0.3 * np.sin(4 * np.pi * f0 * t + phase)).astype(np.float32)


@pytest.fixture
def synthetic_subject(tmp_path):
    """Writes checkpoints/subjects/<name>/ files for one fake subject; returns (root, name)."""
    from src.preprocessing import save_subject

    name = "subjectX"
    rng = np.random.default_rng(0)
    bvp = make_bvp(N_FRAMES, FPS, HR_BPM)
    # ROI: constant skin colour + tiny pulse-locked brightness modulation + noise
    base = np.full((N_FRAMES, 64, 64, 3), 128.0, dtype=np.float32)
    base += 4.0 * bvp[:, None, None, None]
    base += rng.normal(0, 2.0, base.shape)
    roi = np.clip(base, 0, 255).astype(np.uint8)
    data = {
        "roi": roi,
        "bvp": bvp,
        "hr": np.full(N_FRAMES, HR_BPM, dtype=np.float32),
        "ts": (np.arange(N_FRAMES) / FPS).astype(np.float32),
        "meta": {
            "subject": name,
            "n_frames": N_FRAMES,
            "fps": FPS,
            "duration_s": (N_FRAMES - 1) / FPS,
            "face_box_xywh": [0, 0, 64, 64],
            "frame_size_hw": [64, 64],
            "roi_size_hw": [64, 64],
            "channel_order": "BGR",
            "gt_resampled_to_frames": False,
        },
    }
    root = str(tmp_path / "subjects")
    save_subject(data, root)
    assert os.path.exists(os.path.join(root, name, "meta.json"))
    return root, name

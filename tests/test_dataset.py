import numpy as np
import torch

from src import config
from src.dataset import WindowDataset, augment_clip, normalize_clip, normalize_target
from src.preprocessing import load_subject, subject_is_processed
from tests.conftest import FPS, HR_BPM, N_FRAMES


def test_normalize_clip_temporal_removes_dc():
    rng = np.random.default_rng(0)
    x = rng.uniform(0.2, 0.8, size=(3, 150, 8, 8)).astype(np.float32)
    x += np.linspace(0, 0.3, 3)[:, None, None, None]  # per-channel DC offset
    y = normalize_clip(x, "temporal")
    assert y.shape == x.shape
    assert np.abs(y.mean(axis=1)).max() < 1e-4  # per-pixel temporal mean is zero
    assert abs(y.std() - 1.0) < 1e-3  # window-global unit std
    assert np.array_equal(normalize_clip(x, "raw"), x)


def test_normalize_target():
    y = normalize_target(np.arange(150, dtype=np.float32) * 3 + 10)
    assert abs(y.mean()) < 1e-5 and abs(y.std() - 1) < 1e-3


def test_augment_clip_shape_range_and_time_constancy():
    rng = np.random.default_rng(0)
    x = rng.uniform(0.3, 0.7, size=(150, 64, 64, 3)).astype(np.float32)
    x[:, 10:20, 10:20, :] = 0.9  # a bright patch to make the shift detectable
    y = augment_clip(x.copy(), np.random.default_rng(1))
    assert y.shape == x.shape
    assert y.min() >= 0.0 and y.max() <= 1.0
    # The transform is the same for every frame: frame-to-frame differences are unchanged
    d_before = x[1:] - x[:-1]
    d_after = y[1:] - y[:-1]
    # ratio of temporal-difference energies equals the (constant) gain*contrast factor squared
    ratio = (d_after**2).sum() / (d_before**2).sum()
    assert 0.6 < ratio < 1.5


def test_window_dataset_items(synthetic_subject):
    root, name = synthetic_subject
    assert subject_is_processed(name, root)
    store = load_subject(name, root)
    assert store["roi"].shape == (N_FRAMES, 64, 64, 3) and store["roi"].dtype == np.uint8

    ds = WindowDataset([name], window_frames=150, stride=150, normalize="temporal", subject_root=root)
    assert len(ds) == 3
    assert ds.windows_per_subject() == {name: 3}
    x, y, info = ds[1]
    assert isinstance(x, torch.Tensor) and x.shape == (3, 150, 64, 64) and x.dtype == torch.float32
    assert y.shape == (150,) and abs(float(y.mean())) < 1e-4 and abs(float(y.std()) - 1) < 1e-2
    assert info["subject"] == name and info["start"] == 150 and info["fps"] == FPS
    assert abs(info["hr_ref"] - HR_BPM) < 1e-6
    assert info["mean_frame"].shape == (3, 64, 64)
    assert 0.3 < float(info["mean_frame"].mean()) < 0.7  # appearance kept in [0,1]
    assert abs(float(x.mean(dim=1).abs().max())) < 1e-3  # DC removed from x


def test_window_dataset_stride_and_augment(synthetic_subject):
    root, name = synthetic_subject
    ds = WindowDataset([name], window_frames=150, stride=30, augment=True, subject_root=root, seed=3)
    assert len(ds) == len(range(0, N_FRAMES - 150 + 1, 30))
    x1, _, _ = ds[0]
    x2, _, _ = ds[0]
    assert x1.shape == x2.shape == (3, 150, 64, 64)
    assert not torch.allclose(x1, x2)  # augmentation is random


def test_window_dataset_skips_short_subject(synthetic_subject, capsys):
    root, name = synthetic_subject
    ds = WindowDataset([name], window_frames=N_FRAMES + 1, stride=1, subject_root=root)
    assert len(ds) == 0 and ds.subjects == []

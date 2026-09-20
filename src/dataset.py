"""
Window-level PyTorch Dataset built on the per-subject preprocessed ROI sequences.

A window is `window_frames` consecutive frames of one subject starting at `start`.
Windows are enumerated with a `stride` (overlapping for training, non-overlapping for
evaluation) and cut on the fly from the memory-mapped uint8 ROI array, so:

  * stride / window length / augmentation / normalization are training-time choices
  * RAM use is one window at a time, not the whole split

Each item is (x, y, info):
    x     float32 (3, T, H, W)   normalized ROI clip
    y     float32 (T,)           zero-mean / unit-variance BVP target
    info  dict                   subject, start, fps, hr_ref (mean oximeter HR over the window),
                                 mean_frame float32 (3, H, W): the window's temporal mean in [0, 1]
                                 (the appearance that temporal normalization removes from x)
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from . import config
from .preprocessing import load_subject


# ==============================================================================
# INPUT NORMALIZATION
# ==============================================================================
def normalize_clip(x: np.ndarray, mode: str = config.NORMALIZE) -> np.ndarray:
    """
    x: float32 (C, T, H, W) in [0, 1].

    raw      - unchanged (legacy behaviour).
    temporal - subtract each pixel's mean over time (removes the skin-colour DC term and
               static illumination), then divide by ONE global std for the window. Using a
               global rather than per-pixel std keeps the relative AC amplitude between
               skin pixels (pulse + noise) and background pixels (noise only), so the
               network can still tell them apart.
    """
    if mode == "raw":
        return x
    if mode == "temporal":
        x = x - x.mean(axis=1, keepdims=True)
        return x / (x.std() + config.TEMPORAL_NORM_EPS)
    raise ValueError(f"Unknown normalize mode: {mode!r} (expected 'raw' or 'temporal')")


def normalize_target(y: np.ndarray, eps: float = config.EPSILON) -> np.ndarray:
    return ((y - y.mean()) / (y.std() + eps)).astype(np.float32)


# ==============================================================================
# AUGMENTATION
# ==============================================================================
def augment_clip(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    x: float32 (T, H, W, C) in [0, 1]. Every transform uses ONE random draw for the whole
    window so the augmentation is constant over time and cannot inject a fake pulse.
    """
    # Horizontal flip
    if rng.random() < config.AUG_HFLIP_P:
        x = x[:, :, ::-1, :]

    # Brightness (gain) + contrast around the window mean
    gain = 1.0 + rng.uniform(-config.AUG_BRIGHTNESS, config.AUG_BRIGHTNESS)
    contrast = 1.0 + rng.uniform(-config.AUG_CONTRAST, config.AUG_CONTRAST)
    mean = x.mean()
    x = (x - mean) * contrast + mean
    x = x * gain

    # Spatial shift (same for all frames), edge padding
    s = config.AUG_MAX_SHIFT
    if s > 0:
        dy, dx = rng.integers(-s, s + 1, size=2)
        h, w = x.shape[1], x.shape[2]
        x = np.pad(x, ((0, 0), (s, s), (s, s), (0, 0)), mode="edge")
        x = x[:, s + dy : s + dy + h, s + dx : s + dx + w, :]

    return np.clip(x, 0.0, 1.0)


# ==============================================================================
# DATASET
# ==============================================================================
class WindowDataset(Dataset):
    def __init__(
        self,
        subjects: Sequence[str],
        window_frames: int = config.WINDOW_FRAMES,
        stride: int = config.EVAL_STRIDE,
        normalize: str = config.NORMALIZE,
        augment: bool = False,
        subject_root: str = config.SUBJECT_DIR,
        seed: int = config.RANDOM_SEED,
    ):
        self.window_frames = window_frames
        self.stride = stride
        self.normalize = normalize
        self.augment = augment
        self.rng = np.random.default_rng(seed)

        self.subjects: List[str] = []
        self.stores: List[Dict] = []
        self.index: List[Tuple[int, int]] = []  # (subject_idx, start_frame)

        for name in subjects:
            store = load_subject(name, subject_root, mmap=True)
            n = int(store["meta"]["n_frames"])
            if n < window_frames:
                print(f"  [{name}] only {n} frames < window {window_frames} - skipped")
                continue
            si = len(self.stores)
            self.subjects.append(name)
            self.stores.append(store)
            for start in range(0, n - window_frames + 1, stride):
                self.index.append((si, start))

    def __len__(self) -> int:
        return len(self.index)

    def windows_per_subject(self) -> Dict[str, int]:
        counts: Dict[str, int] = {s: 0 for s in self.subjects}
        for si, _ in self.index:
            counts[self.subjects[si]] += 1
        return counts

    def __getitem__(self, i: int):
        si, start = self.index[i]
        store = self.stores[si]
        end = start + self.window_frames

        clip = np.asarray(store["roi"][start:end], dtype=np.float32) / 255.0  # (T, H, W, C)
        if self.augment:
            clip = augment_clip(clip, self.rng)
        x = np.ascontiguousarray(np.transpose(clip, (3, 0, 1, 2)))  # (C, T, H, W)
        mean_frame = np.ascontiguousarray(x.mean(axis=1))  # (C, H, W), before normalization
        x = normalize_clip(x, self.normalize).astype(np.float32)

        y = normalize_target(store["bvp"][start:end])

        info = {
            "subject": self.subjects[si],
            "start": int(start),
            "fps": float(store["meta"]["fps"]),
            "hr_ref": float(store["hr"][start:end].mean()),
            "mean_frame": torch.from_numpy(mean_frame),
        }
        return torch.from_numpy(x), torch.from_numpy(y), info


def _worker_init(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    if info is not None:
        ds = info.dataset
        ds.rng = np.random.default_rng(config.RANDOM_SEED + 1000 * worker_id + info.seed % 1000)


# ==============================================================================
# LOADERS
# ==============================================================================
def make_train_loader(
    subjects: Sequence[str] = config.TRAIN_SUBJECTS,
    batch_size: int = config.BATCH_SIZE,
    stride: int = config.TRAIN_STRIDE,
    normalize: str = config.NORMALIZE,
    augment: bool = True,
    windows_per_epoch: Optional[int] = config.WINDOWS_PER_EPOCH,
    seed: int = config.RANDOM_SEED,
    num_workers: int = 0,
) -> DataLoader:
    ds = WindowDataset(subjects, stride=stride, normalize=normalize, augment=augment, seed=seed)
    gen = torch.Generator().manual_seed(seed)
    if windows_per_epoch is not None and windows_per_epoch < len(ds):
        sampler = RandomSampler(ds, replacement=False, num_samples=windows_per_epoch, generator=gen)
    else:
        sampler = RandomSampler(ds, generator=gen)
    return DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        worker_init_fn=_worker_init if num_workers > 0 else None,
        drop_last=False,
    )


def make_eval_loader(
    subjects: Sequence[str],
    batch_size: int = config.BATCH_SIZE,
    stride: int = config.EVAL_STRIDE,
    normalize: str = config.NORMALIZE,
    num_workers: int = 0,
) -> DataLoader:
    ds = WindowDataset(subjects, stride=stride, normalize=normalize, augment=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


# ==============================================================================
# LEGACY (pre-cut .npy windows from the original pipeline) — sanity checks only
# ==============================================================================
class LegacyValDataset(Dataset):
    """
    The original pipeline's rppg_X_val.npy / rppg_y_val.npy (71 raw-pixel windows).
    Used once to verify the new metrics reproduce the numbers of the old checkpoint.
    """

    def __init__(self, checkpoint_dir: str = config.CHECKPOINT_DIR):
        import os

        self.X = np.load(os.path.join(checkpoint_dir, "rppg_X_val.npy"), mmap_mode="r")
        self.y = np.load(os.path.join(checkpoint_dir, "rppg_y_val.npy"))
        self.subjects = np.load(os.path.join(checkpoint_dir, "val_subjects_used.npy"))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        x = torch.from_numpy(np.ascontiguousarray(self.X[i])).float()
        y = torch.from_numpy(self.y[i]).float()
        info = {
            "subject": str(self.subjects[i]),
            "start": -1,
            "fps": 30.0,
            "hr_ref": float("nan"),
            "mean_frame": x.mean(dim=1),
        }
        return x, y, info

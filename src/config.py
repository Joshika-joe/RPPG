"""
Project configuration: paths, preprocessing constants, windowing, and training defaults.

Subject splits are read from splits/*.txt (single source of truth).
"""

import os
import random
from typing import List

import numpy as np
import torch

# ==============================================================================
# PATHS
# ==============================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATASET_ROOT = os.path.join(PROJECT_ROOT, "DATASET_2")
SPLITS_DIR = os.path.join(PROJECT_ROOT, "splits")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
SUBJECT_DIR = os.path.join(CHECKPOINT_DIR, "subjects")  # per-subject preprocessed ROI sequences
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Checkpoint of the v2 model trained on the legacy (non-overlapping, raw-pixel) pipeline.
LEGACY_MODEL_V2_PATH = os.path.join(PROJECT_ROOT, "best_rppg_model_v2.pth")

# ==============================================================================
# PREPROCESSING
# ==============================================================================
TARGET_SIZE = (64, 64)  # (H, W) of the face ROI
INPUT_CHANNELS = 3

# Haar cascade parameters
FACE_SCALE_FACTOR = 1.1
FACE_MIN_NEIGHBORS = 5
FACE_MIN_SIZE = (50, 50)
FACE_DETECT_FRAMES = 30  # frames used to estimate one fixed face box per subject

# ==============================================================================
# WINDOWING
# ==============================================================================
WINDOW_FRAMES = 150  # ~5 s at ~30 fps
TRAIN_STRIDE = 30  # overlapping windows for training (1 s hop)
EVAL_STRIDE = 150  # non-overlapping windows for val/test
CONTINUOUS_STRIDE = 30  # sliding stride for overlap-add inference
SEGMENT_FRAMES = 300  # ~10 s segments for continuous-inference metrics
EPSILON = 1e-8  # target normalization

# Input normalization mode:
#   "raw"      - pixels / 255 (what the legacy v2 checkpoint was trained on)
#   "temporal" - per-pixel temporal mean removed, then scaled by the window's global std.
#                Removes the skin-colour DC term so the ~1% pulse component dominates the input.
NORMALIZE = "temporal"
TEMPORAL_NORM_EPS = 1e-6

# Augmentation (train only). All are applied identically to every frame of a window so
# they cannot inject a spurious temporal rhythm.
AUG_HFLIP_P = 0.5
AUG_BRIGHTNESS = 0.10  # multiplicative gain in [1-b, 1+b]
AUG_CONTRAST = 0.10  # contrast factor in [1-c, 1+c]
AUG_MAX_SHIFT = 4  # pixels, random spatial shift (edge-padded)

# ==============================================================================
# METRICS
# ==============================================================================
HR_BAND_HZ = (0.7, 4.0)  # 42–240 bpm; plausible human HR range for FFT peak search
HR_TOLERANCE_BPM = 5.0
HR_SUBHARMONIC_RATIO = 0.5  # prefer a peak near f/2 over the top FFT peak f if its power is >= this fraction
HR_SUBHARMONIC_TOL = 0.12  # relative half-width of the search band around f/2

# ==============================================================================
# TRAINING DEFAULTS
# ==============================================================================
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 50
PATIENCE = 8
PEARSON_MSE_WEIGHT = 0.2  # MSE weight inside the 'pearson' loss
WINDOWS_PER_EPOCH = 400  # random subset of the overlapping windows seen per epoch (CPU budget)
RANDOM_SEED = 42


# ==============================================================================
# HELPERS
# ==============================================================================
def load_split(name: str) -> List[str]:
    """Reads splits/<name>_subjects.txt -> list of subject folder names."""
    path = os.path.join(SPLITS_DIR, f"{name}_subjects.txt")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


TRAIN_SUBJECTS = load_split("train")
VAL_SUBJECTS = load_split("val")
TEST_SUBJECTS = load_split("test")
ALL_SUBJECTS = TRAIN_SUBJECTS + VAL_SUBJECTS + TEST_SUBJECTS

SPLITS = {"train": TRAIN_SUBJECTS, "val": VAL_SUBJECTS, "test": TEST_SUBJECTS}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

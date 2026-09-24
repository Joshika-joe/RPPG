import numpy as np
import pytest

from src.live import GRID_FPS, WINDOW_SECONDS, ChestMotion, FaceTracker, find_peaks, resample_clip
from tests.conftest import make_bvp


def test_grid_constants():
    assert WINDOW_SECONDS == 5.0
    assert GRID_FPS == 30.0


def test_resample_clip_recovers_a_known_timeline():
    """Frames encode their own timestamp as brightness; resampling must recover it linearly."""
    cap_fps = 17.0  # awkward rate, nothing like the model's 30
    times = np.arange(0, 6.0, 1.0 / cap_fps)
    frames = [np.full((4, 4, 3), min(255, t * 40), dtype=np.uint8) for t in times]
    target = np.linspace(1.0, 6.0 - 1.0 / cap_fps, 150)
    clip = resample_clip(frames, times, target)
    assert clip.shape == (150, 4, 4, 3)
    assert clip.dtype == np.float32
    got = clip[:, 0, 0, 0] * 255.0
    expected = np.minimum(255.0, target * 40)
    assert np.abs(got - expected).max() < 2.0  # uint8 quantization only


def test_resample_clip_is_frame_rate_independent():
    """The same underlying signal sampled at 30 and at 15 fps resamples to the same clip."""
    dur = 6.0
    def build(fps):
        t = np.arange(0, dur, 1.0 / fps)
        sig = 128 + 100 * np.sin(2 * np.pi * 1.2 * t)  # 72 bpm
        return [np.full((2, 2, 3), v, dtype=np.uint8) for v in sig], t

    f30, t30 = build(30.0)
    f15, t15 = build(15.0)
    target = np.linspace(1.0, 5.0, 150)
    a = resample_clip(f30, t30, target)[:, 0, 0, 0]
    b = resample_clip(f15, t15, target)[:, 0, 0, 0]
    assert np.corrcoef(a, b)[0, 1] > 0.999
    assert np.abs(a - b).max() < 0.05  # 15 fps interpolation error on a 1.2 Hz signal


def test_find_peaks_counts_beats():
    fs, hr = 30.0, 72.0
    sig = make_bvp(int(20 * fs), fs, hr)
    peaks = find_peaks(sig, fs)
    expected = 20 * hr / 60.0
    assert abs(len(peaks) - expected) <= 2
    assert np.all(np.diff(peaks) >= int(fs * 60.0 / 200.0))


def test_find_peaks_empty_on_short_input():
    assert find_peaks(np.array([1.0, 2.0]), 30.0).size == 0


def test_chest_motion_needs_data_before_reporting():
    cm = ChestMotion()
    assert np.isnan(cm.breathing_rate())


def test_chest_roi_below_face_clips_to_frame():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    assert ChestMotion.roi_below_face(frame, (40, 10, 60, 60)) is not None
    assert ChestMotion.roi_below_face(frame, (40, 110, 60, 60)) is None  # nothing below


def test_face_tracker_keeps_box_between_detections():
    tr = FaceTracker(every=1000)
    tr.box = (10, 20, 30, 30)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    assert tr.update(frame) == (10, 20, 30, 30)  # no detection run yet

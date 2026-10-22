"""
Live-pipeline gating: timing gaps, face-box changes, overlap-add holes, rejection of
unreliable estimates, and equivalence between the offline and live preprocessing paths.
"""

import numpy as np
import torch

from src import config
from src.dataset import WindowDataset, normalize_clip
from src.inference import overlap_add
from src.live import (
    GRID_FPS,
    STATE_FILLING,
    STATE_LOW_SIGNAL,
    STATE_OK,
    STATE_TIMING,
    WINDOW_SECONDS,
    LiveEstimator,
    resample_clip,
)
from tests.conftest import FPS, make_bvp

W = config.WINDOW_FRAMES


class PulseModel(torch.nn.Module):
    """Stands in for the network: returns the mean brightness track, which is the injected pulse."""

    def forward(self, x, appearance=None):
        return x.mean(dim=(1, 3, 4))


def make_estimator(**kw):
    kw.setdefault("hr_seconds", 10.0)
    kw.setdefault("resp_seconds", 45.0)
    return LiveEstimator(PulseModel(), "temporal", torch.device("cpu"), **kw)


def feed(est, seconds, fps, hr_bpm=72.0, t0=1000.0, gap_at=None, gap=0.0, box=(0, 0, 64, 64),
         noise=0.0, seed=0):
    """Pushes synthetic face crops whose brightness carries a pulse at hr_bpm."""
    rng = np.random.default_rng(seed)
    t = t0
    n = int(seconds * fps)
    for i in range(n):
        if gap_at is not None and i == gap_at:
            t += gap
        phase = 2 * np.pi * (hr_bpm / 60.0) * (t - t0)
        v = 128 + 40 * np.sin(phase)
        frame = np.full((64, 64, 3), np.clip(v, 0, 255), dtype=np.uint8)
        if noise:
            frame = np.clip(frame + rng.normal(0, noise, frame.shape), 0, 255).astype(np.uint8)
        est.push(frame, t, box=box)
        t += 1.0 / fps


# ==============================================================================
# TIMING
# ==============================================================================
def test_window_spans_an_exact_sampling_grid():
    """150 samples at 30 Hz span 149 intervals, not 150 - otherwise HR is biased +0.67%."""
    assert abs(WINDOW_SECONDS - (W - 1) / GRID_FPS) < 1e-12
    est = make_estimator()
    feed(est, 8.0, 30.0)
    clip, _ = est.snapshot()
    assert clip.shape == (W, 64, 64, 3)


def test_insufficient_frames_state():
    est = make_estimator()
    feed(est, 1.0, 30.0)
    assert est.snapshot() is None
    assert est.state == STATE_FILLING
    assert not est.ready()


def test_timestamp_gap_is_rejected_not_interpolated():
    """A dropped-frame gap inside the window must not be papered over by interpolation."""
    est = make_estimator(max_gap_seconds=0.25)
    feed(est, 8.0, 30.0, gap_at=180, gap=1.0)  # one-second stall inside the last window
    assert est.snapshot() is None
    assert est.state == STATE_TIMING
    assert est.max_gap > 0.25


def test_small_jitter_is_tolerated():
    est = make_estimator(max_gap_seconds=0.25)
    feed(est, 8.0, 30.0, gap_at=180, gap=0.05)
    assert est.snapshot() is not None
    assert est.state != STATE_TIMING


def test_irregular_timestamps_still_resample_correctly():
    """Variable frame intervals must not change the recovered rate."""
    est = make_estimator()
    rng = np.random.default_rng(0)
    t, hr = 500.0, 84.0
    for _ in range(int(9 * 25)):
        v = 128 + 40 * np.sin(2 * np.pi * (hr / 60.0) * (t - 500.0))
        est.push(np.full((64, 64, 3), np.clip(v, 0, 255), dtype=np.uint8), t, box=(0, 0, 64, 64))
        t += max(0.01, rng.normal(1 / 25.0, 0.008))  # jittery 25 fps
    snap = est.snapshot()
    assert snap is not None
    est.infer_window(*snap)
    assert abs(est.hr - hr) < 6.0, est.hr


# ==============================================================================
# FACE BOX
# ==============================================================================
def test_face_box_change_clears_the_buffer():
    """A re-snapped crop is a different geometry; a window must not span the change."""
    est = make_estimator()
    feed(est, 8.0, 30.0, box=(0, 0, 64, 64))
    assert est.ready()
    est.push(np.full((64, 64, 3), 128, dtype=np.uint8), 2000.0, box=(10, 10, 70, 70))
    assert not est.ready()
    assert len(est.frames) == 1


def test_same_box_does_not_clear_the_buffer():
    est = make_estimator()
    feed(est, 8.0, 30.0, box=(5, 5, 60, 60))
    n = len(est.frames)
    est.push(np.full((64, 64, 3), 128, dtype=np.uint8), 2000.0, box=(5, 5, 60, 60))
    assert len(est.frames) >= n  # appended, not reset


# ==============================================================================
# OVERLAP-ADD HOLES
# ==============================================================================
def test_overlap_add_reports_coverage():
    out, covered = overlap_add([(0, np.ones(W)), (600, np.ones(W))], 600 + W, W,
                               return_coverage=True)
    assert covered[:W].all() and covered[600:].all()
    assert not covered[W:600].any()
    assert np.all(out[W:600] == 0)


def test_hole_in_the_signal_is_reported_as_a_timing_problem():
    est = make_estimator()
    est.t0 = 0.0
    est.windows.append((0, make_bvp(W, FPS, 72.0)))
    est.windows.append((900, make_bvp(W, FPS, 72.0)))  # 30 s later, nothing in between
    est.update_estimates()
    assert est.state == STATE_TIMING
    assert est.coverage < 0.999
    assert est.rejects == 1


# ==============================================================================
# REJECTION OF UNRELIABLE ESTIMATES
# ==============================================================================
def test_clean_signal_is_accepted():
    est = make_estimator(min_quality=-3.0, min_peak_margin=-999)
    feed(est, 20.0, 30.0, hr_bpm=72.0)
    for _ in range(3):
        snap = est.snapshot()
        est.infer_window(*snap)
    assert est.state == STATE_OK
    assert est.signal_ok
    assert abs(est.hr - 72.0) < 5.0
    assert est.availability == 1.0


def test_disagreement_between_estimators_rejects():
    """Two estimators that disagree means neither can be trusted."""
    est = make_estimator(max_agreement_bpm=0.0, min_peak_margin=-999)  # force disagreement
    feed(est, 20.0, 30.0, hr_bpm=72.0)
    snap = est.snapshot()
    est.infer_window(*snap)
    assert est.state == STATE_LOW_SIGNAL
    assert not est.signal_ok
    assert est.availability == 0.0


def test_quality_floor_rejects_noise():
    est = make_estimator(min_quality=20.0)  # unreachable threshold
    feed(est, 20.0, 30.0, hr_bpm=72.0, noise=20.0, seed=5)
    snap = est.snapshot()
    est.infer_window(*snap)
    assert not est.signal_ok
    assert est.rejects >= 1


def test_availability_is_reported():
    est = make_estimator()
    assert not np.isfinite(est.availability)  # nothing attempted yet
    feed(est, 20.0, 30.0)
    est.infer_window(*est.snapshot())
    assert 0.0 <= est.availability <= 1.0


# ==============================================================================
# OFFLINE / LIVE CONSISTENCY
# ==============================================================================
def test_live_resampling_reproduces_the_offline_clip(synthetic_subject):
    """
    The same frames, timestamped at the dataset's own rate, must reach the model identically
    through the offline WindowDataset and the live buffer. This is the check that training
    and inference see the same thing.
    """
    root, name = synthetic_subject
    ds = WindowDataset([name], window_frames=W, stride=W, normalize="temporal",
                       subject_root=root)
    x_offline, _, info = ds[0]

    store = ds.stores[0]
    est = make_estimator()
    for i in range(W):
        est.push(np.asarray(store["roi"][i]), i / FPS, box=(0, 0, 64, 64))
    snap = est.snapshot()
    assert snap is not None, "live buffer rejected a clean dataset-rate stream"
    clip, _ = snap
    x_live = normalize_clip(
        np.ascontiguousarray(np.transpose(clip, (3, 0, 1, 2))), "temporal"
    ).astype(np.float32)

    assert x_live.shape == tuple(x_offline.shape)
    # The live path resamples onto a 30 Hz grid while the subject is 30 fps, so the two
    # should agree to interpolation error only.
    assert np.abs(x_live - x_offline.numpy()).max() < 0.05
    assert abs(float(np.corrcoef(x_live.ravel(), x_offline.numpy().ravel())[0, 1]) - 1.0) < 1e-3
    assert info["fps"] == FPS

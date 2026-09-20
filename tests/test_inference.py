import numpy as np
import torch

from src.dataset import WindowDataset
from src.inference import hann_weights, overlap_add, predict_continuous, segment_signals
from tests.conftest import FPS, HR_BPM, N_FRAMES, make_bvp


def test_hann_weights_strictly_positive():
    w = hann_weights(150)
    assert w.shape == (150,)
    assert w.min() > 0
    assert abs(w.max() - 1.0) < 1e-3  # even N: peak falls between samples


def test_overlap_add_reconstructs_signal():
    n, win, stride = 450, 150, 30
    sig = make_bvp(n, FPS, HR_BPM)
    starts = list(range(0, n - win + 1, stride))
    windows = [(s, 2.0 * sig[s : s + win] + 7.0) for s in starts]  # arbitrary per-window scale/offset
    out = overlap_add(windows, n, win)
    assert out.shape == (n,)
    assert np.isfinite(out).all()
    r = np.corrcoef(out, sig)[0, 1]
    assert r > 0.999


def test_overlap_add_handles_uncovered_tail():
    n, win, stride = 200, 150, 40  # starts 0, 40 -> frames 190..199 uncovered
    windows = [(s, np.ones(win)) for s in (0, 40)]
    out = overlap_add(windows, n, win)
    assert np.isfinite(out).all()
    assert np.all(out[190:] == 0)


def test_window_dataset_cover_tail(synthetic_subject):
    root, name = synthetic_subject
    ds = WindowDataset([name], window_frames=150, stride=100, subject_root=root, cover_tail=False)
    ds_tail = WindowDataset([name], window_frames=150, stride=100, subject_root=root, cover_tail=True)
    # 450 frames, stride 100 -> starts 0,100,200,300 ; tail start 300 already ends on last frame
    assert [s for _, s in ds.index] == [0, 100, 200, 300]
    assert [s for _, s in ds_tail.index] == [0, 100, 200, 300]
    ds_tail2 = WindowDataset([name], window_frames=150, stride=120, subject_root=root, cover_tail=True)
    assert [s for _, s in ds_tail2.index] == [0, 120, 240, 300]


class _Identity(torch.nn.Module):
    """Returns the mean over channels and pixels of each frame -> follows the injected pulse."""

    def forward(self, x, appearance=None):
        return x.mean(dim=(1, 3, 4))


def test_predict_continuous_and_segment(synthetic_subject, monkeypatch):
    root, name = synthetic_subject
    import src.inference as inf

    # point the dataset at the temp root
    orig = inf.WindowDataset
    monkeypatch.setattr(inf, "WindowDataset", lambda *a, **k: orig(*a, subject_root=root, **k))

    signals = predict_continuous(_Identity(), [name], stride=30, normalize="temporal", device=torch.device("cpu"))
    sig = signals[name]
    assert sig["pred"].shape == (N_FRAMES,)
    assert sig["n_windows"] == len(range(0, N_FRAMES - 150 + 1, 30))
    assert np.corrcoef(sig["pred"], sig["gt"])[0, 1] > 0.9

    preds, targets, fps, subj, hr_ref = segment_signals(signals, segment_frames=150)
    assert preds.shape == targets.shape == (3, 150)
    assert np.allclose(preds.mean(1), 0, atol=1e-5) and np.allclose(preds.std(1), 1, atol=1e-3)
    assert list(fps) == [FPS] * 3 and subj == [name] * 3
    assert np.allclose(hr_ref, HR_BPM)

import numpy as np
import pytest

from src import config
from src.metrics import (
    bandpass_filter,
    compute_metrics,
    estimate_hr_fft,
    pearson_per_window,
    snr_db,
)
from tests.conftest import make_bvp


def sine(hr_bpm, fps=30.0, n=150, phase=0.0):
    t = np.arange(n) / fps
    return np.sin(2 * np.pi * hr_bpm / 60.0 * t + phase)


@pytest.mark.parametrize("hr", [45.0, 72.0, 110.0, 150.0])
def test_hr_fft_recovers_sine(hr):
    assert abs(estimate_hr_fft(sine(hr), 30.0) - hr) < 1.0


def test_hr_fft_uses_fps():
    # Same samples interpreted at 23.3 fps -> HR scales by 23.3/30
    sig = sine(90.0, fps=30.0)
    assert abs(estimate_hr_fft(sig, 23.3) - 90.0 * 23.3 / 30.0) < 1.0


def test_hr_fft_prefers_fundamental_over_dominant_harmonic():
    # A peaky pulse whose 2nd harmonic out-powers the fundamental
    t = np.arange(150) / 30.0
    f0 = 100.0 / 60.0
    sig = 0.8 * np.sin(2 * np.pi * f0 * t) + 1.0 * np.sin(4 * np.pi * f0 * t)
    naive = estimate_hr_fft(sig, 30.0, subharmonic_ratio=0.0)
    fixed = estimate_hr_fft(sig, 30.0)
    assert abs(naive - 200.0) < 2.0
    assert abs(fixed - 100.0) < 2.0


def test_hr_fft_does_not_halve_a_clean_sine():
    assert abs(estimate_hr_fft(sine(130.0), 30.0) - 130.0) < 1.0


def test_bandpass_keeps_pulse_removes_drift_and_noise():
    fps, n = 30.0, 300
    t = np.arange(n) / fps
    pulse = np.sin(2 * np.pi * 1.2 * t)
    drift = 3.0 * np.sin(2 * np.pi * 0.1 * t)
    hf = 0.8 * np.sin(2 * np.pi * 8.0 * t)
    out = bandpass_filter(pulse + drift + hf, fps)
    r = np.corrcoef(out, pulse)[0, 1]
    assert r > 0.98
    assert np.abs(out).max() < 1.5


def test_pearson_per_window():
    x = np.random.default_rng(0).normal(size=(3, 150))
    r = pearson_per_window(x, x)
    assert np.allclose(r, 1.0, atol=1e-6)
    assert np.allclose(pearson_per_window(-x, x), -1.0, atol=1e-6)
    assert np.allclose(pearson_per_window(3 * x + 5, x), 1.0, atol=1e-6)  # scale/offset invariant


def test_snr_clean_vs_noise():
    # On a 5-s rectangular window spectral leakage caps a pure tone at ~6 dB; the metric is
    # only meaningful relative to other signals scored the same way.
    clean = sine(72.0)
    noise = np.random.default_rng(1).normal(size=150)
    s_clean, s_noise = snr_db(clean, 72.0, 30.0), snr_db(noise, 72.0, 30.0)
    assert s_clean > 5.0
    assert s_noise < 0.0
    assert s_clean - s_noise > 5.0


def test_compute_metrics_perfect_and_shifted():
    fps = 30.0
    hrs = [60.0, 80.0, 100.0]
    targets = np.stack([make_bvp(150, fps, h) for h in hrs])
    targets = (targets - targets.mean(1, keepdims=True)) / targets.std(1, keepdims=True)
    subjects = ["a", "a", "b"]

    m = compute_metrics(targets, targets, fps=fps, subjects=subjects, hr_ref=np.array(hrs))
    s = m["summary"]
    assert s["n_windows"] == 3
    assert s["mse"] < 1e-10
    assert s["pearson_mean"] > 0.999
    assert s["hr_mae_bpm"] < 1.0
    assert s["hr_mae_vs_oximeter_bpm"] < 1.0
    assert s[f"hr_within_{int(config.HR_TOLERANCE_BPM)}bpm_frac"] == 1.0
    assert abs(s["amplitude_ratio_mean"] - 1.0) < 1e-6
    assert set(m["per_subject"]) == {"a", "b"}
    assert m["per_subject"]["a"]["n_windows"] == 2

    # Half-amplitude prediction: same r and HR, amplitude ratio 0.5, MSE > 0
    m2 = compute_metrics(0.5 * targets, targets, fps=fps)
    assert m2["summary"]["pearson_mean"] > 0.999
    assert m2["summary"]["hr_mae_bpm"] < 1.0
    assert abs(m2["summary"]["amplitude_ratio_mean"] - 0.5) < 1e-6
    assert m2["summary"]["mse"] > 0.1

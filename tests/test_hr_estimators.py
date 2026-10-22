"""Deterministic tests on synthetic waveforms with known rates (not real-world validation)."""

import numpy as np
import pytest

from src import config
from src.hr_estimators import (
    METHODS,
    estimate_hr,
    estimate_hr_detail,
    peak_margin_db,
    welch_psd,
)


def pulse(hr_bpm, fs=30.0, seconds=10.0, harmonic=0.3, noise=0.0, seed=0):
    """Peaked pulse-like waveform: fundamental plus a second harmonic."""
    t = np.arange(int(seconds * fs)) / fs
    f0 = hr_bpm / 60.0
    x = np.sin(2 * np.pi * f0 * t) + harmonic * np.sin(4 * np.pi * f0 * t)
    if noise:
        x = x + np.random.default_rng(seed).normal(0, noise, size=len(t))
    return x


def quantisation_bpm(hr: float, fs: float = 30.0) -> float:
    """
    Beat counting resolves the period only to whole samples, so neighbouring periods are
    hr^2/(60*fs) bpm apart - about 10 bpm at 132 bpm and 30 fps. Spectral and autocorrelation
    estimators interpolate between bins and are not limited this way.
    """
    return hr * hr / (60.0 * fs)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("hr", [48.0, 72.0, 96.0, 132.0])
def test_every_estimator_recovers_a_clean_sinusoid(method, hr):
    tol = max(2.5, 1.1 * quantisation_bpm(hr)) if method == "peaks" else 2.5
    got = estimate_hr(pulse(hr, harmonic=0.0), 30.0, method=method)
    assert abs(got - hr) < tol, f"{method} gave {got:.1f} for {hr} (tol {tol:.1f})"


@pytest.mark.parametrize("method", ["fft", "welch", "autocorr"])
@pytest.mark.parametrize("hr", [48.0, 72.0, 96.0, 132.0])
def test_spectral_and_autocorr_handle_a_realistic_pulse_shape(method, hr):
    """A pulse with a moderate second harmonic - the normal case for real BVP."""
    got = estimate_hr(pulse(hr, harmonic=0.3), 30.0, method=method)
    assert abs(got - hr) < 2.5, f"{method} gave {got:.1f} for {hr}"


@pytest.mark.parametrize("method", METHODS)
def test_frequency_to_bpm_scales_with_sampling_rate(method):
    """The same samples at half the rate describe half the heart rate."""
    sig = pulse(90.0, fs=30.0, seconds=12.0, harmonic=0.0)
    assert abs(estimate_hr(sig, 30.0, method=method) - 90.0) < 2.5
    assert abs(estimate_hr(sig, 15.0, method=method) - 45.0) < 2.0


# ------------------------------------------------------------------ harmonic confusion
# Measured behaviour when the SECOND HARMONIC carries more power than the fundamental
# (power ratio = fundamental power / harmonic power):
#
#   autocorr   returns the fundamental at every ratio - no doubling failure mode
#   fft/welch  return double until the ratio reaches HR_SUBHARMONIC_RATIO (0.5), then flip
#   peaks      counts harmonic peaks as beats until the fundamental dominates (~1.4)
#
# This is why autocorr is the live default: the spectral correction depends on a threshold
# that has to be chosen, and below it the error is a clean factor of two.
def harmonic_case(fundamental_amp, hr=100.0, seconds=10.0, fs=30.0):
    t = np.arange(int(seconds * fs)) / fs
    f0 = hr / 60.0
    return fundamental_amp * np.sin(2 * np.pi * f0 * t) + 1.0 * np.sin(4 * np.pi * f0 * t)


def test_autocorr_is_immune_to_a_dominant_second_harmonic():
    for amp in (0.5, 0.7, 0.9):
        got = estimate_hr(harmonic_case(amp), 30.0, method="autocorr")
        assert abs(got - 100.0) < 6.0, f"amp {amp} -> {got:.1f}"


@pytest.mark.parametrize("method", ["fft", "welch"])
def test_spectral_correction_fires_above_its_threshold_and_not_below(method):
    ratio = config.HR_SUBHARMONIC_RATIO
    just_below = estimate_hr(harmonic_case(np.sqrt(ratio) - 0.02), 30.0, method=method)
    just_above = estimate_hr(harmonic_case(np.sqrt(ratio) + 0.02), 30.0, method=method)
    assert abs(just_below - 200.0) < 6.0, f"{method} below threshold gave {just_below:.1f}"
    assert abs(just_above - 100.0) < 6.0, f"{method} above threshold gave {just_above:.1f}"


def test_subharmonic_correction_is_load_bearing_for_fft():
    sig = harmonic_case(0.9)
    assert abs(estimate_hr(sig, 30.0, method="fft", subharmonic_ratio=0.0) - 200.0) < 6.0
    assert abs(estimate_hr(sig, 30.0, method="fft") - 100.0) < 6.0


def test_peak_counting_miscounts_when_the_harmonic_dominates():
    """Documents a known limitation: beat counting is unsuitable on harmonic-heavy waveforms."""
    assert abs(estimate_hr(harmonic_case(0.7), 30.0, method="peaks") - 200.0) < 8.0


@pytest.mark.parametrize("method", METHODS)
def test_no_spurious_halving_on_a_pure_tone(method):
    """The sub-harmonic rule must not fire when there is nothing at f/2."""
    tol = max(3.0, 1.1 * quantisation_bpm(140.0)) if method == "peaks" else 3.0
    got = estimate_hr(pulse(140.0, harmonic=0.0), 30.0, method=method)
    assert abs(got - 140.0) < tol, f"{method} gave {got:.1f}"
    assert got > 100.0, f"{method} halved a clean tone to {got:.1f}"


@pytest.mark.parametrize("method", METHODS)
def test_insufficient_duration_returns_nan_or_stays_sane(method):
    """Half a second of signal cannot support a heart-rate estimate."""
    got = estimate_hr(pulse(72.0, seconds=0.5, harmonic=0.0), 30.0, method=method)
    assert (not np.isfinite(got)) or config.HR_BAND_HZ[0] * 60 <= got <= config.HR_BAND_HZ[1] * 60


@pytest.mark.parametrize("method", METHODS)
def test_estimate_is_always_inside_the_search_band_or_nan(method):
    rng = np.random.default_rng(3)
    for _ in range(5):
        got = estimate_hr(rng.normal(size=300), 30.0, method=method)
        assert (not np.isfinite(got)) or (
            config.HR_BAND_HZ[0] * 60 - 1 <= got <= config.HR_BAND_HZ[1] * 60 + 1
        )


@pytest.mark.parametrize("method", METHODS)
def test_constant_and_empty_input_do_not_crash(method):
    assert not np.isfinite(estimate_hr(np.zeros(300), 30.0, method=method))
    assert not np.isfinite(estimate_hr(np.ones(300), 30.0, method=method))


def test_noise_degrades_agreement_between_estimators():
    """Estimator agreement is the quality signal used live - it must react to noise."""
    clean = pulse(72.0, harmonic=0.0, noise=0.0)
    noisy = np.random.default_rng(1).normal(size=300)
    d_clean = abs(estimate_hr(clean, 30.0, "autocorr") - estimate_hr(clean, 30.0, "fft"))
    d_noise = abs(estimate_hr(noisy, 30.0, "autocorr") - estimate_hr(noisy, 30.0, "fft"))
    assert d_clean < 2.0
    assert d_noise > d_clean


def test_peak_margin_is_large_for_a_tone_and_small_for_noise():
    tone = estimate_hr_detail(pulse(72.0, harmonic=0.0), 30.0, method="fft")
    noise = estimate_hr_detail(np.random.default_rng(2).normal(size=300), 30.0, method="fft")
    assert peak_margin_db(tone) > peak_margin_db(noise)


def test_welch_averages_segments_and_falls_back_when_short():
    f_long, p_long = welch_psd(pulse(72.0, seconds=20.0, harmonic=0.0), 30.0, segment_seconds=4.0)
    assert len(f_long) == len(p_long) and p_long.max() > 0
    peak = f_long[np.argmax(p_long)] * 60
    assert abs(peak - 72.0) < 3.0
    f_short, p_short = welch_psd(pulse(72.0, seconds=1.0, harmonic=0.0), 30.0, segment_seconds=4.0)
    assert len(f_short) == len(p_short)  # single-periodogram fallback, no crash


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        estimate_hr(pulse(72.0, harmonic=0.0), 30.0, method="nope")

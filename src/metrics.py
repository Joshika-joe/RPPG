"""
rPPG evaluation metrics.

MSE alone is a poor metric for BVP estimation: a model that predicts zeros scores 1.0 on
unit-variance targets, and a model that gets the rhythm right but the amplitude wrong is
penalised heavily. What the literature reports instead:

    Pearson r     waveform shape agreement per window (scale-invariant)
    HR MAE/RMSE   heart-rate error in bpm from the dominant FFT peak
    SNR           power at the reference HR (+ 1st harmonic) vs the rest of the band (de Haan)

All functions take numpy arrays. `fps` may be a scalar or a per-window array.
"""

from typing import Dict, Optional, Sequence, Union

import numpy as np

from . import config

ArrayLike = Union[float, np.ndarray, Sequence[float]]


# ==============================================================================
# SIGNAL HELPERS
# ==============================================================================
def bandpass_filter(
    sig: np.ndarray,
    fs: float,
    band=config.HR_BAND_HZ,
    transition_hz: float = 0.1,
) -> np.ndarray:
    """
    Zero-phase FFT-domain band-pass with raised-cosine edges (numpy only; scipy.signal is
    blocked by Application Control on some Windows machines). sig: (T,) or (N, T), one fs.
    """
    sig = np.asarray(sig, dtype=np.float64)
    n = sig.shape[-1]
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    lo, hi = band
    gain = np.ones_like(freqs)
    gain[freqs < lo - transition_hz] = 0.0
    gain[freqs > hi + transition_hz] = 0.0
    ramp_lo = (freqs >= lo - transition_hz) & (freqs < lo)
    ramp_hi = (freqs > hi) & (freqs <= hi + transition_hz)
    gain[ramp_lo] = 0.5 * (1 - np.cos(np.pi * (freqs[ramp_lo] - (lo - transition_hz)) / transition_hz))
    gain[ramp_hi] = 0.5 * (1 + np.cos(np.pi * (freqs[ramp_hi] - hi) / transition_hz))
    mean = sig.mean(axis=-1, keepdims=True)
    out = np.fft.irfft(np.fft.rfft(sig - mean, axis=-1) * gain, n=n, axis=-1)
    return out.astype(np.float32)


def _spectrum(sig: np.ndarray, fs: float, nfft_factor: int = 8):
    sig = sig - sig.mean()
    n = int(len(sig) * nfft_factor)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power = np.abs(np.fft.rfft(sig, n=n)) ** 2
    return freqs, power


def estimate_hr_fft(
    sig: np.ndarray,
    fs: float,
    band=config.HR_BAND_HZ,
    nfft_factor: int = 8,
    subharmonic_ratio: float = config.HR_SUBHARMONIC_RATIO,
) -> float:
    """
    HR in bpm from the dominant spectral peak inside `band`.

    Peaky BVP waveforms can put more power in the 2nd harmonic than in the fundamental, so
    after finding the top peak f we also look at f/2: if there is a peak there with at least
    `subharmonic_ratio` of the top peak's power, f/2 is the heart rate.
    """
    freqs, power = _spectrum(sig, fs, nfft_factor)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not mask.any():
        return float("nan")
    fb, pb = freqs[mask], power[mask]
    i_max = int(np.argmax(pb))
    f_max, p_max = fb[i_max], pb[i_max]

    f_half = f_max / 2.0
    if subharmonic_ratio > 0 and f_half >= band[0]:
        near = np.abs(fb - f_half) <= config.HR_SUBHARMONIC_TOL * f_half  # relative band around f/2
        if near.any():
            j = int(np.flatnonzero(near)[np.argmax(pb[near])])
            is_local_peak = (j == 0 or pb[j] >= pb[j - 1]) and (j == len(pb) - 1 or pb[j] >= pb[j + 1])
            if is_local_peak and pb[j] >= subharmonic_ratio * p_max:
                return float(fb[j] * 60.0)
    return float(f_max * 60.0)


def snr_db(
    sig: np.ndarray,
    hr_bpm: float,
    fs: float,
    band=config.HR_BAND_HZ,
    tol_hz: float = 0.1,
    nfft_factor: int = 8,
) -> float:
    """
    de Haan SNR: power within ±tol_hz of the reference HR frequency and its first harmonic,
    over the remaining power inside `band`.

    Spectral leakage of the rectangular window caps this at ~6 dB for a perfect 5-s sine,
    so treat it as a relative score between models, not an absolute quality.
    """
    if not np.isfinite(hr_bpm):
        return float("nan")
    freqs, power = _spectrum(sig, fs, nfft_factor)
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    f0 = hr_bpm / 60.0
    signal_mask = (np.abs(freqs - f0) <= tol_hz) | (np.abs(freqs - 2 * f0) <= tol_hz)
    signal_mask &= in_band
    noise_mask = in_band & ~signal_mask
    s = power[signal_mask].sum()
    n = power[noise_mask].sum()
    if n <= 0 or s <= 0:
        return float("nan")
    return float(10.0 * np.log10(s / n))


def pearson_per_window(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation. pred, target: (N, T)."""
    p = pred - pred.mean(axis=1, keepdims=True)
    t = target - target.mean(axis=1, keepdims=True)
    num = (p * t).sum(axis=1)
    den = np.sqrt((p**2).sum(axis=1) * (t**2).sum(axis=1)) + 1e-12
    return num / den


# ==============================================================================
# AGGREGATE
# ==============================================================================
def compute_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    fps: ArrayLike = 30.0,
    hr_ref: Optional[np.ndarray] = None,
    subjects: Optional[Sequence[str]] = None,
    bandpass: bool = False,
    tolerance_bpm: float = config.HR_TOLERANCE_BPM,
) -> Dict:
    """
    preds, targets : (N, T) float arrays (targets are zero-mean / unit-variance).
    fps            : scalar or (N,) per-window frame rate.
    hr_ref         : optional (N,) reference HR from the oximeter (bpm).
    subjects       : optional (N,) subject names -> adds a per-subject table.
    bandpass       : band-pass the prediction (0.7-4 Hz) before Pearson / HR / SNR.

    Returns {"summary": {...}, "per_window": {...}, "per_subject": {...}}.
    """
    preds = np.asarray(preds, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    n = len(preds)
    fps_arr = np.broadcast_to(np.asarray(fps, dtype=np.float64), (n,)).copy()

    if bandpass:
        preds = np.stack([bandpass_filter(p, f) for p, f in zip(preds, fps_arr)])

    mse_pw = ((preds - targets) ** 2).mean(axis=1)
    r_pw = pearson_per_window(preds, targets)
    hr_pred = np.array([estimate_hr_fft(p, f) for p, f in zip(preds, fps_arr)])
    hr_gt = np.array([estimate_hr_fft(t, f) for t, f in zip(targets, fps_arr)])
    hr_err = np.abs(hr_pred - hr_gt)
    snr_pw = np.array([snr_db(p, h, f) for p, h, f in zip(preds, hr_gt, fps_arr)])
    amp_ratio = preds.std(axis=1) / (targets.std(axis=1) + 1e-12)

    summary = {
        "n_windows": int(n),
        "mse": float(mse_pw.mean()),
        "pearson_mean": float(np.nanmean(r_pw)),
        "pearson_median": float(np.nanmedian(r_pw)),
        "pearson_frac_gt_0.5": float((r_pw > 0.5).mean()),
        "hr_mae_bpm": float(np.nanmean(hr_err)),
        "hr_rmse_bpm": float(np.sqrt(np.nanmean(hr_err**2))),
        f"hr_within_{int(tolerance_bpm)}bpm_frac": float((hr_err <= tolerance_bpm).mean()),
        "snr_db_mean": float(np.nanmean(snr_pw)),
        "amplitude_ratio_mean": float(amp_ratio.mean()),
        "bandpass": bool(bandpass),
    }

    per_window = {
        "mse": mse_pw.tolist(),
        "pearson": r_pw.tolist(),
        "hr_pred_bpm": hr_pred.tolist(),
        "hr_gt_bpm": hr_gt.tolist(),
        "hr_abs_err_bpm": hr_err.tolist(),
        "snr_db": snr_pw.tolist(),
        "amplitude_ratio": amp_ratio.tolist(),
        "fps": fps_arr.tolist(),
    }

    if hr_ref is not None:
        hr_ref = np.asarray(hr_ref, dtype=np.float64)
        valid = np.isfinite(hr_ref)
        if valid.any():
            err_ref = np.abs(hr_pred - hr_ref)
            summary["hr_mae_vs_oximeter_bpm"] = float(np.nanmean(err_ref[valid]))
            summary["hr_gt_fft_vs_oximeter_mae_bpm"] = float(
                np.nanmean(np.abs(hr_gt - hr_ref)[valid])
            )
            per_window["hr_ref_bpm"] = hr_ref.tolist()

    result = {"summary": summary, "per_window": per_window}

    if subjects is not None:
        subjects = np.asarray(subjects)
        per_subject = {}
        for s in sorted(set(subjects.tolist())):
            m = subjects == s
            per_subject[s] = {
                "n_windows": int(m.sum()),
                "mse": float(mse_pw[m].mean()),
                "pearson_mean": float(np.nanmean(r_pw[m])),
                "hr_mae_bpm": float(np.nanmean(hr_err[m])),
                "snr_db_mean": float(np.nanmean(snr_pw[m])),
                "amplitude_ratio_mean": float(amp_ratio[m].mean()),
            }
        result["per_subject"] = per_subject
        per_window["subject"] = subjects.tolist()

    return result


def format_summary(summary: Dict) -> str:
    keys = [
        ("mse", "MSE", "{:.4f}"),
        ("pearson_mean", "Pearson r (mean)", "{:.3f}"),
        ("pearson_median", "Pearson r (median)", "{:.3f}"),
        ("hr_mae_bpm", "HR MAE", "{:.2f} bpm"),
        ("hr_rmse_bpm", "HR RMSE", "{:.2f} bpm"),
        (f"hr_within_{int(config.HR_TOLERANCE_BPM)}bpm_frac", "HR within tol", "{:.1%}"),
        ("hr_mae_vs_oximeter_bpm", "HR MAE vs oximeter", "{:.2f} bpm"),
        ("snr_db_mean", "SNR", "{:.2f} dB"),
        ("amplitude_ratio_mean", "Amplitude ratio", "{:.2f}"),
    ]
    lines = [f"  windows: {summary.get('n_windows')}"]
    for key, label, fmt in keys:
        if key in summary:
            lines.append(f"  {label:<20} {fmt.format(summary[key])}")
    return "\n".join(lines)

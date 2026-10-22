"""
Alternative heart-rate estimators, all with the same signature so they can be swapped and
compared on identical inputs.

    hr_bpm = estimate_hr(sig, fs, method="welch")

Every estimator takes a band-passed, zero-mean waveform and returns beats per minute, or NaN
when it cannot produce an estimate. Each also exposes a *_detail variant returning the
candidate peaks, which the quality logic and the failure-analysis tool use.

Implemented with numpy only - scipy.signal cannot be imported on some Windows machines
(Application Control blocks its compiled extensions).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config


# ==============================================================================
# SHARED SPECTRAL HELPERS
# ==============================================================================
def _band_mask(freqs: np.ndarray, band) -> np.ndarray:
    return (freqs >= band[0]) & (freqs <= band[1])


def _local_peaks(power: np.ndarray) -> np.ndarray:
    """Indices of strict local maxima (interior points only)."""
    if len(power) < 3:
        return np.array([], dtype=int)
    return np.flatnonzero((power[1:-1] > power[:-2]) & (power[1:-1] >= power[2:])) + 1


def _refine_peak(freqs: np.ndarray, power: np.ndarray, i: int) -> float:
    """Quadratic interpolation around bin i - sub-bin accuracy without zero padding."""
    if i <= 0 or i >= len(power) - 1:
        return float(freqs[i])
    y0, y1, y2 = power[i - 1], power[i], power[i + 1]
    denom = y0 - 2 * y1 + y2
    if denom == 0:
        return float(freqs[i])
    delta = 0.5 * (y0 - y2) / denom
    delta = float(np.clip(delta, -1.0, 1.0))
    return float(freqs[i] + delta * (freqs[1] - freqs[0]))


def periodogram(sig: np.ndarray, fs: float, nfft_factor: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Single-window periodogram with a Hann taper (reduces the leakage a boxcar leaves)."""
    sig = np.asarray(sig, dtype=np.float64)
    sig = (sig - sig.mean()) * np.hanning(len(sig))
    n = int(len(sig) * nfft_factor)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power = np.abs(np.fft.rfft(sig, n=n)) ** 2
    return freqs, power


def welch_psd(
    sig: np.ndarray, fs: float, segment_seconds: float = 4.0, overlap: float = 0.5,
    nfft_factor: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Welch's method: average periodograms of overlapping Hann-tapered segments. Trades
    frequency resolution for variance, so the dominant peak is less likely to be a noise spike.
    Falls back to a single periodogram when the signal is shorter than one segment.
    """
    sig = np.asarray(sig, dtype=np.float64)
    seg_len = int(segment_seconds * fs)
    if seg_len < 16 or len(sig) < seg_len:
        return periodogram(sig, fs, nfft_factor)
    step = max(1, int(seg_len * (1.0 - overlap)))
    window = np.hanning(seg_len)
    nfft = int(seg_len * nfft_factor)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    acc = np.zeros(len(freqs))
    count = 0
    for start in range(0, len(sig) - seg_len + 1, step):
        seg = sig[start : start + seg_len]
        seg = (seg - seg.mean()) * window
        acc += np.abs(np.fft.rfft(seg, n=nfft)) ** 2
        count += 1
    return freqs, acc / max(count, 1)


# ==============================================================================
# ESTIMATORS
# ==============================================================================
def _from_spectrum(
    freqs: np.ndarray, power: np.ndarray, band, subharmonic_ratio: float, subharmonic_tol: float,
    refine: bool,
) -> Dict:
    """Shared peak selection, including the sub-harmonic (frequency-doubling) correction."""
    mask = _band_mask(freqs, band)
    if not mask.any() or power[mask].max() <= 0:
        return {"hr_bpm": float("nan"), "peaks": [], "corrected": False}
    fb, pb = freqs[mask], power[mask]
    i_max = int(np.argmax(pb))
    f_sel, corrected = fb[i_max], False

    f_half = fb[i_max] / 2.0
    if subharmonic_ratio > 0 and f_half >= band[0]:
        near = np.abs(fb - f_half) <= subharmonic_tol * f_half
        if near.any():
            idx = np.flatnonzero(near)
            j = int(idx[np.argmax(pb[idx])])
            is_peak = (j == 0 or pb[j] >= pb[j - 1]) and (j == len(pb) - 1 or pb[j] >= pb[j + 1])
            if is_peak and pb[j] >= subharmonic_ratio * pb[i_max]:
                f_sel, i_max, corrected = fb[j], j, True

    if refine:
        f_sel = _refine_peak(fb, pb, i_max)

    order = _local_peaks(pb)
    order = order[np.argsort(pb[order])[::-1]][:4] if len(order) else np.array([], dtype=int)
    peaks = [(float(fb[k] * 60.0), float(pb[k] / pb.max())) for k in order]
    return {"hr_bpm": float(f_sel * 60.0), "peaks": peaks, "corrected": corrected}


def estimate_hr_fft_detail(sig, fs, band=config.HR_BAND_HZ, nfft_factor: int = 8,
                           subharmonic_ratio: float = config.HR_SUBHARMONIC_RATIO,
                           subharmonic_tol: float = config.HR_SUBHARMONIC_TOL,
                           refine: bool = False) -> Dict:
    freqs, power = periodogram(sig, fs, nfft_factor)
    return _from_spectrum(freqs, power, band, subharmonic_ratio, subharmonic_tol, refine)


def estimate_hr_welch_detail(sig, fs, band=config.HR_BAND_HZ, segment_seconds: float = 4.0,
                             subharmonic_ratio: float = config.HR_SUBHARMONIC_RATIO,
                             subharmonic_tol: float = config.HR_SUBHARMONIC_TOL,
                             refine: bool = False) -> Dict:
    freqs, power = welch_psd(sig, fs, segment_seconds=segment_seconds)
    return _from_spectrum(freqs, power, band, subharmonic_ratio, subharmonic_tol, refine)


def estimate_hr_autocorr_detail(sig, fs, band=config.HR_BAND_HZ) -> Dict:
    """
    Autocorrelation: the first strong peak at a lag inside the plausible HR range gives the
    period directly. Immune to the frequency-doubling failure of spectral peak picking,
    because a half-period peak is always weaker than the full-period one for a real pulse.
    """
    x = np.asarray(sig, dtype=np.float64)
    x = x - x.mean()
    if x.std() == 0 or len(x) < 16:
        return {"hr_bpm": float("nan"), "peaks": [], "corrected": False}
    n = int(2 ** np.ceil(np.log2(2 * len(x))))
    spec = np.fft.rfft(x, n=n)
    ac = np.fft.irfft(spec * np.conj(spec), n=n)[: len(x)]
    ac /= ac[0] if ac[0] != 0 else 1.0

    lag_min = int(np.ceil(fs / band[1]))  # fastest allowed HR -> shortest lag
    lag_max = int(np.floor(fs / band[0]))
    lag_max = min(lag_max, len(ac) - 2)
    if lag_max <= lag_min + 1:
        return {"hr_bpm": float("nan"), "peaks": [], "corrected": False}

    seg = ac[lag_min : lag_max + 1]
    cand = _local_peaks(seg)
    if len(cand) == 0:
        return {"hr_bpm": float("nan"), "peaks": [], "corrected": False}
    best = int(cand[np.argmax(seg[cand])])
    lag = _refine_peak(np.arange(len(seg), dtype=float), seg, best) + lag_min
    order = cand[np.argsort(seg[cand])[::-1]][:4]
    peaks = [(float(60.0 * fs / (k + lag_min)), float(seg[k])) for k in order]
    return {"hr_bpm": float(60.0 * fs / lag), "peaks": peaks, "corrected": False}


def estimate_hr_peaks_detail(sig, fs, band=config.HR_BAND_HZ, prominence: float = 0.3) -> Dict:
    """
    Time-domain beat counting: median inter-beat interval of detected systolic peaks.
    Only meaningful when the waveform is clean; reports NaN when too few beats are found.
    """
    x = np.asarray(sig, dtype=np.float64)
    x = x - x.mean()
    if x.std() == 0:
        return {"hr_bpm": float("nan"), "peaks": [], "corrected": False}
    thr = prominence * x.std()
    cand = _local_peaks(x)
    cand = cand[x[cand] > thr]
    min_dist = max(1, int(fs / band[1]))
    kept: List[int] = []
    for i in cand[np.argsort(x[cand])[::-1]]:
        if all(abs(i - k) >= min_dist for k in kept):
            kept.append(int(i))
    kept.sort()
    if len(kept) < 4:
        return {"hr_bpm": float("nan"), "peaks": [], "corrected": False}
    ibi = np.diff(np.array(kept)) / fs
    ibi = ibi[(ibi >= 1.0 / band[1]) & (ibi <= 1.0 / band[0])]
    if len(ibi) < 3:
        return {"hr_bpm": float("nan"), "peaks": [], "corrected": False}
    hr = 60.0 / float(np.median(ibi))
    spread = float(np.std(60.0 / ibi))
    return {"hr_bpm": hr, "peaks": [(hr, 1.0)], "corrected": False, "beat_hr_std": spread}


# ==============================================================================
# DISPATCH
# ==============================================================================
DETAIL_FUNCS = {
    "fft": estimate_hr_fft_detail,
    "welch": estimate_hr_welch_detail,
    "autocorr": estimate_hr_autocorr_detail,
    "peaks": estimate_hr_peaks_detail,
}
METHODS = list(DETAIL_FUNCS)


def estimate_hr_detail(sig, fs, method: str = "fft", **kwargs) -> Dict:
    if method not in DETAIL_FUNCS:
        raise ValueError(f"Unknown HR method {method!r}. Available: {METHODS}")
    return DETAIL_FUNCS[method](sig, fs, **kwargs)


def estimate_hr(sig, fs, method: str = "fft", **kwargs) -> float:
    return estimate_hr_detail(sig, fs, method=method, **kwargs)["hr_bpm"]


def peak_margin_db(detail: Dict) -> float:
    """
    How far the chosen peak stands above the next competing peak, in dB. A small margin means
    two candidate rates are nearly tied, which is where doubling/halving errors come from.
    """
    peaks = detail.get("peaks") or []
    if len(peaks) < 2:
        return float("inf") if peaks else float("nan")
    powers = sorted((p for _, p in peaks), reverse=True)
    if powers[1] <= 0:
        return float("inf")
    return float(10.0 * np.log10(powers[0] / powers[1]))

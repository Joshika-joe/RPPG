"""Window-duration comparison machinery (Phase 7D). Synthetic signals only."""

import numpy as np
import pytest

from src import config
from src.compare_windows import (
    DURATIONS,
    GATE,
    HOP_SECONDS,
    evaluate_duration,
    gate_decision,
    hr_at,
    nonstationarity,
    reference_hr_series,
    summarise,
)
from tests.conftest import make_bvp

FPS = 30.0


def signal(hr_bpm=72.0, seconds=40.0, fps=FPS):
    return make_bvp(int(seconds * fps), fps, hr_bpm)


# ==============================================================================
# hr_at
# ==============================================================================
def test_hr_at_reads_the_window_ending_at_the_given_index():
    sig = signal(72.0)
    out = hr_at(sig, FPS, end_idx=int(20 * FPS), n=int(10 * FPS))
    assert abs(out["hr"] - 72.0) < 2.0
    assert np.isfinite(out["snr"]) and np.isfinite(out["margin"])


def test_hr_at_returns_nan_when_the_window_is_not_fully_available():
    sig = signal(72.0, seconds=6.0)
    out = hr_at(sig, FPS, end_idx=int(3 * FPS), n=int(10 * FPS))
    assert not np.isfinite(out["hr"])


def test_hr_at_returns_nan_on_a_constant_window():
    out = hr_at(np.zeros(900), FPS, end_idx=600, n=300)
    assert not np.isfinite(out["hr"])


def test_longer_windows_resolve_frequency_more_finely():
    """Documents the physical limit: raw bin width is 60/W bpm."""
    for dur in DURATIONS:
        assert 60.0 / dur == pytest.approx({5.0: 12.0, 8.0: 7.5, 10.0: 6.0, 15.0: 4.0}[dur])


# ==============================================================================
# QUALITY GATE
# ==============================================================================
def test_gate_accepts_a_clean_window():
    assert gate_decision({"hr": 72.0, "snr": 6.0, "margin": 9.0, "cross": 72.5}) == []


def test_gate_names_each_failing_check():
    assert "snr" in gate_decision({"hr": 72.0, "snr": -9.0, "margin": 9.0, "cross": 72.0})
    assert "margin" in gate_decision({"hr": 72.0, "snr": 6.0, "margin": 0.1, "cross": 72.0})
    assert "disagreement" in gate_decision({"hr": 72.0, "snr": 6.0, "margin": 9.0, "cross": 95.0})


def test_gate_reports_every_reason_not_just_the_first():
    reasons = gate_decision({"hr": 72.0, "snr": -9.0, "margin": 0.1, "cross": 95.0})
    assert set(reasons) == {"snr", "margin", "disagreement"}


def test_gate_rejects_an_invalid_estimate_without_further_checks():
    assert gate_decision({"hr": float("nan"), "snr": 9.0, "margin": 9.0, "cross": 72.0}) == ["invalid"]


def test_gate_thresholds_are_the_documented_validation_values():
    assert GATE["min_snr_db"] == -1.9
    assert GATE["min_peak_margin_db"] == 3.2
    assert GATE["max_disagreement_bpm"] == 5.7


# ==============================================================================
# NON-STATIONARITY
# ==============================================================================
def test_reference_series_uses_a_fixed_probe_regardless_of_window():
    """The probe must be duration-independent so non-stationarity is measured identically."""
    series = reference_hr_series(signal(72.0, seconds=30.0), FPS)
    vals = [v for v in series.values() if np.isfinite(v)]
    assert len(vals) > 10
    assert all(abs(v - 72.0) < 3.0 for v in vals)


def test_nonstationarity_is_zero_for_a_constant_rate():
    series = reference_hr_series(signal(72.0, seconds=30.0), FPS)
    spread = nonstationarity(series, end_idx=int(25 * FPS), n=int(10 * FPS), fps=FPS)
    assert spread < 3.0


def test_nonstationarity_detects_a_rate_change():
    fps = FPS
    sig = np.concatenate([make_bvp(int(15 * fps), fps, 60.0), make_bvp(int(15 * fps), fps, 110.0)])
    series = reference_hr_series(sig, fps)
    spread = nonstationarity(series, end_idx=int(25 * fps), n=int(20 * fps), fps=fps)
    assert spread > 20.0


def test_nonstationarity_is_nan_without_enough_probes():
    assert not np.isfinite(nonstationarity({}, end_idx=300, n=150, fps=FPS))


# ==============================================================================
# PROTOCOLS
# ==============================================================================
def _signals(hr=72.0, seconds=40.0):
    sig = signal(hr, seconds)
    return {"subjX": {"pred": sig.copy(), "gt": sig.copy(), "fps": FPS, "n_windows": 1}}


def test_sliding_protocol_uses_identical_end_times_for_every_duration():
    """The primary comparison must not differ in which intervals each duration covers."""
    ends = {}
    for dur in DURATIONS:
        res = evaluate_duration(_signals(), dur, "sliding")
        ends[dur] = [round(r["end_s"], 3) for r in res["records"]]
    reference = ends[DURATIONS[0]]
    for dur in DURATIONS:
        assert ends[dur] == reference, f"{dur}s evaluated at different timestamps"
    assert len(reference) > 5


def test_block_protocol_yields_fewer_segments_for_longer_windows():
    counts = {d: len(evaluate_duration(_signals(), d, "blocks")["records"]) for d in DURATIONS}
    assert counts[5.0] > counts[10.0] > counts[15.0]


def test_evaluate_duration_recovers_a_known_rate():
    res = evaluate_duration(_signals(hr=84.0), 10.0, "sliding")
    errs = [r["err"] for r in res["records"] if np.isfinite(r["err"])]
    assert len(errs) > 5
    assert np.mean(errs) < 2.0
    assert res["seconds_per_estimate"] > 0


# ==============================================================================
# SUMMARY
# ==============================================================================
def _records(errs, accepted=True, subject="s1"):
    return [{"subject": subject, "end_s": float(i), "hr_pred": 72.0 + e, "hr_ref": 72.0,
             "err": e, "snr": 5.0, "margin": 9.0, "cross": 72.0, "pearson": 0.9,
             "ref_spread": 2.0, "accepted": accepted, "reasons": [] if accepted else ["snr"]}
            for i, e in enumerate(errs)]


def test_summarise_computes_error_statistics():
    s = summarise(_records([0.0, 1.0, 2.0, 3.0]), 10.0, 0.01)
    assert s["mae"] == pytest.approx(1.5)
    assert s["median_abs_err"] == pytest.approx(1.5)
    assert s["rmse"] == pytest.approx(np.sqrt(np.mean([0, 1, 4, 9])))
    assert s["within_5bpm"] == 1.0
    assert s["n_estimates"] == 4


def test_summarise_separates_accepted_from_rejected():
    recs = _records([0.5, 0.5], accepted=True) + _records([20.0], accepted=False)
    s = summarise(recs, 10.0, 0.01)
    assert s["accepted_fraction"] == pytest.approx(2 / 3)
    assert s["mae_accepted"] == pytest.approx(0.5)
    assert s["mae_rejected"] == pytest.approx(20.0)
    assert s["rejection_reasons"] == {"snr": 1}


def test_summarise_reports_latency_resolution_and_update_interval_separately():
    s = summarise(_records([1.0, 1.0]), 15.0, 0.02)
    assert s["first_estimate_latency_s"] == 15.0
    assert s["update_interval_s"] == HOP_SECONDS
    assert s["freq_resolution_bpm"] == pytest.approx(4.0)


def test_summarise_subject_level_mae_does_not_weight_by_window_count():
    many = _records([2.0] * 20, subject="busy")
    few = _records([0.0] * 2, subject="quiet")
    s = summarise(many + few, 10.0, 0.01)
    assert s["mae"] == pytest.approx(np.mean([2.0] * 20 + [0.0] * 2))  # window-weighted
    assert s["subject_level_mae"] == pytest.approx(1.0)  # subject-weighted
    assert s["worst_subject_mae"] == pytest.approx(2.0)
    assert s["n_subjects"] == 2


def test_summarise_stability_is_consecutive_change():
    recs = _records([0.0, 2.0, 4.0])  # hr 72, 74, 76 -> deltas of 2
    s = summarise(recs, 10.0, 0.01)
    assert s["stability_bpm"] == pytest.approx(2.0)

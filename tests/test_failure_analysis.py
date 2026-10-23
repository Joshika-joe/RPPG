"""Failure categorisation, subject-level resampling, and the split/reference audits."""

import numpy as np
import pytest

from src import config
from src.audit_data import audit_splits
from src.failure_analysis import bootstrap_subject_ci, categorise


# ==============================================================================
# CATEGORISATION
# ==============================================================================
def test_categorise_ok_within_tolerance():
    assert categorise(pred=72.0, ref=74.0, err=2.0, snr=5.0, stability=1.0) == "ok"


def test_categorise_doubling_and_halving():
    assert categorise(pred=144.0, ref=72.0, err=72.0, snr=5.0, stability=1.0) == "doubled"
    assert categorise(pred=36.0, ref=72.0, err=36.0, snr=5.0, stability=1.0) == "halved"


def test_categorise_harmonics_take_priority_over_direction():
    """A doubled estimate must not be filed merely as 'too high'."""
    assert categorise(pred=150.0, ref=75.0, err=75.0, snr=1.0, stability=0.0) == "doubled"


def test_categorise_unstable_when_the_rate_moves_mid_window():
    assert categorise(pred=108.0, ref=93.0, err=15.0, snr=-3.0, stability=13.0) == "unstable"


def test_categorise_flags_errors_that_survive_a_good_signal():
    assert categorise(pred=95.0, ref=75.0, err=20.0, snr=6.0, stability=1.0) == "wrong_despite_good_signal"


def test_categorise_direction_when_nothing_else_explains_it():
    assert categorise(pred=95.0, ref=75.0, err=20.0, snr=0.0, stability=1.0) == "too_high"
    assert categorise(pred=55.0, ref=75.0, err=20.0, snr=0.0, stability=1.0) == "too_low"


def test_categorise_invalid():
    assert categorise(pred=float("nan"), ref=72.0, err=float("nan"), snr=0.0, stability=0.0) == "invalid"


# ==============================================================================
# SUBJECT-LEVEL CONFIDENCE INTERVALS
# ==============================================================================
def test_bootstrap_resamples_subjects_not_windows():
    """One bad subject among six must widen the interval, not be diluted by its windows."""
    tight = bootstrap_subject_ci({f"s{i}": 0.5 for i in range(6)})
    skewed = bootstrap_subject_ci({**{f"s{i}": 0.5 for i in range(5)}, "s5": 4.25})
    assert tight["ci95"][1] - tight["ci95"][0] < 0.01  # identical subjects -> no spread
    assert skewed["ci95"][1] - skewed["ci95"][0] > 1.0
    assert skewed["worst_subject"] == "s5"
    assert skewed["worst_subject_mae"] == pytest.approx(4.25)
    assert skewed["median_subject_mae"] == pytest.approx(0.5)
    assert skewed["n_subjects"] == 6


def test_bootstrap_mean_is_the_subject_mean_not_the_window_mean():
    ci = bootstrap_subject_ci({"a": 1.0, "b": 3.0})
    assert ci["mean"] == pytest.approx(2.0)
    assert ci["ci95"][0] <= ci["mean"] <= ci["ci95"][1]


def test_bootstrap_handles_a_single_subject():
    ci = bootstrap_subject_ci({"only": 1.5})
    assert ci["n_subjects"] == 1
    assert not np.isfinite(ci["ci95"][0])


def test_bootstrap_is_reproducible():
    a = bootstrap_subject_ci({f"s{i}": float(i) for i in range(5)})
    b = bootstrap_subject_ci({f"s{i}": float(i) for i in range(5)})
    assert a["ci95"] == b["ci95"]


# ==============================================================================
# SPLIT SEPARATION
# ==============================================================================
def test_splits_are_disjoint_and_complete():
    """Subject-level separation is the premise of every reported metric."""
    sets = {k: set(v) for k, v in config.SPLITS.items()}
    assert sets["train"] & sets["val"] == set()
    assert sets["train"] & sets["test"] == set()
    assert sets["val"] & sets["test"] == set()
    for name, subjects in config.SPLITS.items():
        assert len(subjects) == len(set(subjects)), f"duplicates in {name}"


def test_audit_splits_reports_clean_on_the_real_configuration():
    report = audit_splits()
    assert report["clean"], report["findings"]
    assert report["subject_overlap"] == {}
    assert report["window_source_overlap"] == {}

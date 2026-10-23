"""
UBFC-Phys adapter and the cross-dataset evaluation path.

These are SYNTHETIC tests: a generated video with a known pulse injected into its
brightness, and a generated 64 Hz BVP csv. They verify format handling, resampling and
plumbing. They are NOT evidence of cross-dataset generalisation, which requires the real
dataset.
"""

import json
import os

import cv2
import numpy as np
import pytest

from src import config
from src.eval_cross import best_lag_correlation, discover
from src.preprocessing import load_subject, subject_is_processed
from src.ubfc_phys import (
    BVP_HZ,
    NOMINAL_FPS,
    TASKS,
    find_recordings,
    process_recording,
    read_bvp_csv,
    run,
)

FPS = 30.0  # MJPG at 35 fps is not reliably reproduced by every OpenCV build
HR = 72.0


def write_recording(root, participant="s1", task="T1", seconds=8.0, fps=FPS, hr=HR,
                    size=(128, 128), bvp_hz=BVP_HZ, header=False, index_col=False):
    """Creates s<N>/vid_s<N>_T<k>.avi and the matching bvp csv, with a pulse in brightness."""
    folder = os.path.join(root, participant)
    os.makedirs(folder, exist_ok=True)
    n = int(seconds * fps)

    video = os.path.join(folder, f"vid_{participant}_{task}.avi")
    writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert writer.isOpened(), "OpenCV could not open an MJPG writer"
    for i in range(n):
        v = 128 + 40 * np.sin(2 * np.pi * (hr / 60.0) * i / fps)
        frame = np.full((size[1], size[0], 3), np.clip(v, 0, 255), dtype=np.uint8)
        # a bright blob so Haar has something face-like to find is not required: the tests
        # that need a face box inject it directly.
        writer.write(frame)
    writer.release()

    m = int(seconds * bvp_hz)
    rows = []
    if header:
        rows.append(["bvp"] if not index_col else ["idx", "bvp"])
    for i in range(m):
        val = np.sin(2 * np.pi * (hr / 60.0) * i / bvp_hz)
        rows.append([str(i), f"{val:.6f}"] if index_col else [f"{val:.6f}"])
    with open(os.path.join(folder, f"bvp_{participant}_{task}.csv"), "w",
              encoding="utf-8", newline="") as f:
        for r in rows:
            f.write(",".join(r) + "\n")
    return video


# ==============================================================================
# CSV PARSING
# ==============================================================================
def test_read_bvp_plain(tmp_path):
    write_recording(str(tmp_path))
    v = read_bvp_csv(os.path.join(str(tmp_path), "s1", "bvp_s1_T1.csv"))
    assert v.dtype == np.float32 and len(v) == int(8.0 * BVP_HZ)


def test_read_bvp_tolerates_header_and_index_column(tmp_path):
    write_recording(str(tmp_path), header=True, index_col=True)
    v = read_bvp_csv(os.path.join(str(tmp_path), "s1", "bvp_s1_T1.csv"))
    assert len(v) == int(8.0 * BVP_HZ)
    assert np.isfinite(v).all()


def test_read_bvp_rejects_a_file_with_no_numbers(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("alpha\nbeta\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_bvp_csv(str(p))


# ==============================================================================
# DISCOVERY
# ==============================================================================
def test_find_recordings_matches_the_documented_layout(tmp_path):
    root = str(tmp_path)
    for task in TASKS:
        write_recording(root, "s1", task, seconds=6.0)
    write_recording(root, "s2", "T1", seconds=6.0)
    recs = find_recordings(root)
    assert [r["recording"] for r in recs] == ["s1_T1", "s1_T2", "s1_T3", "s2_T1"]
    assert recs[0]["participant"] == "s1" and recs[0]["task"] == "T1"


def test_find_recordings_filters_by_task_and_participant(tmp_path):
    root = str(tmp_path)
    for task in TASKS:
        write_recording(root, "s1", task, seconds=6.0)
        write_recording(root, "s2", task, seconds=6.0)
    assert len(find_recordings(root, tasks=["T1"])) == 2
    assert len(find_recordings(root, participants=["s2"])) == 3


def test_find_recordings_ignores_unrelated_folders(tmp_path):
    root = str(tmp_path)
    write_recording(root, "s1", "T1", seconds=6.0)
    os.makedirs(os.path.join(root, "docs"), exist_ok=True)
    os.makedirs(os.path.join(root, "subject1"), exist_ok=True)  # UBFC-rPPG naming
    assert [r["recording"] for r in find_recordings(root)] == ["s1_T1"]


def test_find_recordings_skips_a_video_without_its_bvp(tmp_path):
    root = str(tmp_path)
    write_recording(root, "s1", "T1", seconds=6.0)
    os.remove(os.path.join(root, "s1", "bvp_s1_T1.csv"))
    assert find_recordings(root) == []


def test_find_recordings_on_a_missing_root_returns_empty():
    assert find_recordings("/definitely/not/here") == []


# ==============================================================================
# PROCESSING
# ==============================================================================
class _AlwaysFace:
    """Stands in for the Haar cascade so the test does not depend on detecting a real face."""

    def detectMultiScale(self, gray, **kwargs):
        h, w = gray.shape[:2]
        return np.array([[w // 4, h // 4, w // 2, h // 2]])

    def empty(self):
        return False


def test_process_recording_builds_the_standard_store(tmp_path, monkeypatch):
    root = str(tmp_path)
    write_recording(root, "s1", "T2", seconds=8.0)
    rec = find_recordings(root)[0]
    data = process_recording(rec, cascade=_AlwaysFace(), n_detect_frames=5)
    assert data is not None
    m = data["meta"]
    assert m["dataset"] == "UBFC-Phys"
    assert m["participant"] == "s1" and m["task"] == "T2" and m["task_name"] == "speech"
    assert data["roi"].shape[1:] == (*config.TARGET_SIZE, 3)
    assert data["roi"].dtype == np.uint8
    assert len(data["bvp"]) == len(data["ts"]) == data["roi"].shape[0]
    assert m["reference_hz"] == BVP_HZ
    assert m["gt_resampled_to_frames"] is True
    # No reference HR channel is published for this dataset.
    assert np.isnan(data["hr"]).all()


def test_bvp_is_resampled_from_64hz_onto_frame_times(tmp_path):
    """The 64 Hz wrist signal must land on frame times without changing the pulse rate."""
    root = str(tmp_path)
    write_recording(root, "s1", "T1", seconds=10.0, hr=90.0)
    rec = find_recordings(root)[0]
    data = process_recording(rec, cascade=_AlwaysFace(), n_detect_frames=5)
    from src.hr_estimators import estimate_hr

    got = estimate_hr(data["bvp"], data["meta"]["fps"], method="fft")
    assert abs(got - 90.0) < 3.0, f"resampled reference reads {got:.1f} bpm"


def test_reference_coverage_is_reported_when_bvp_is_short(tmp_path):
    root = str(tmp_path)
    write_recording(root, "s1", "T1", seconds=10.0)
    # truncate the reference to half its length
    path = os.path.join(root, "s1", "bvp_s1_T1.csv")
    lines = open(path, encoding="utf-8").read().splitlines()
    open(path, "w", encoding="utf-8").write("\n".join(lines[: len(lines) // 2]) + "\n")
    rec = find_recordings(root)[0]
    data = process_recording(rec, cascade=_AlwaysFace(), n_detect_frames=5)
    assert data["meta"]["reference_coverage"] < 0.6


def test_process_recording_skips_when_no_face_is_found(tmp_path):
    class _NoFace:
        def detectMultiScale(self, gray, **kwargs):
            return np.array([])

        def empty(self):
            return False

    root = str(tmp_path)
    write_recording(root, "s1", "T1", seconds=6.0)
    rec = find_recordings(root)[0]
    assert process_recording(rec, cascade=_NoFace(), n_detect_frames=5) is None


def test_process_recording_skips_a_clip_shorter_than_one_model_window(tmp_path):
    root = str(tmp_path)
    write_recording(root, "s1", "T1", seconds=2.0)  # 60 frames < 150
    rec = find_recordings(root)[0]
    assert process_recording(rec, cascade=_AlwaysFace(), n_detect_frames=5) is None


def test_run_writes_a_loadable_store_in_a_separate_root(tmp_path, monkeypatch):
    import src.ubfc_phys as up

    monkeypatch.setattr(up, "get_face_detector", lambda: _AlwaysFace())
    root = str(tmp_path / "raw")
    out = str(tmp_path / "store")
    write_recording(root, "s1", "T1", seconds=8.0)
    write_recording(root, "s1", "T2", seconds=8.0)
    done = run(root, out_root=out)
    assert set(done) == {"s1_T1", "s1_T2"}
    for name in done:
        assert subject_is_processed(name, out)
    store = load_subject("s1_T1", out)
    assert store["meta"]["dataset"] == "UBFC-Phys"
    # the UBFC-rPPG cache must be untouched
    assert out != config.SUBJECT_DIR


def test_run_is_idempotent(tmp_path, monkeypatch):
    import src.ubfc_phys as up

    monkeypatch.setattr(up, "get_face_detector", lambda: _AlwaysFace())
    root, out = str(tmp_path / "raw"), str(tmp_path / "store")
    write_recording(root, "s1", "T1", seconds=8.0)
    assert run(root, out_root=out) == ["s1_T1"]
    assert run(root, out_root=out) == ["s1_T1"]  # second pass uses the cache


def test_run_raises_a_helpful_error_on_an_empty_root(tmp_path):
    with pytest.raises(SystemExit) as e:
        run(str(tmp_path))
    assert "UBFC-Phys" in str(e.value)


# ==============================================================================
# CROSS-DATASET EVALUATION HELPERS
# ==============================================================================
def test_discover_requires_an_adapted_store(tmp_path):
    with pytest.raises(SystemExit) as e:
        discover(str(tmp_path / "missing"))
    assert "src.ubfc_phys" in str(e.value)


def test_best_lag_correlation_finds_a_known_offset():
    """A wrist reference lags the facial pulse; the diagnostic must recover a shift."""
    fps = 30.0
    t = np.arange(int(12 * fps)) / fps
    sig = np.sin(2 * np.pi * 1.2 * t)
    shift = 9  # samples
    out = best_lag_correlation(sig[shift:], sig[: len(sig) - shift], fps)
    assert out["r_best_lag"] > 0.95
    assert out["r_best_lag"] >= out["r_zero_lag"]


def test_best_lag_correlation_is_identity_for_aligned_signals():
    fps = 30.0
    t = np.arange(int(12 * fps)) / fps
    sig = np.sin(2 * np.pi * 1.2 * t)
    out = best_lag_correlation(sig, sig, fps)
    assert out["r_zero_lag"] > 0.999
    assert out["best_lag_s"] == 0.0


def test_nominal_fps_constant_matches_the_documented_rate():
    assert NOMINAL_FPS == 35.0
    assert BVP_HZ == 64.0

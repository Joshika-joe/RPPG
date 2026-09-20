import os

import cv2
import numpy as np
import pytest

from src.preprocessing import (
    crop_resize,
    detect_face_box,
    get_face_detector,
    process_subject,
    read_ground_truth,
)


def test_read_ground_truth(tmp_path):
    p = tmp_path / "ground_truth.txt"
    p.write_text("1.0 2.0 3.0\n70 70 71\n0.0 0.033 0.066\n", encoding="utf-8")
    bvp, hr, ts = read_ground_truth(str(p))
    assert bvp.tolist() == [1.0, 2.0, 3.0]
    assert hr.tolist() == [70.0, 70.0, 71.0]
    assert ts.dtype == np.float32 and len(ts) == 3


def test_read_ground_truth_rejects_mismatch(tmp_path):
    p = tmp_path / "ground_truth.txt"
    p.write_text("1.0 2.0 3.0\n70 70\n0.0 0.033 0.066\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_ground_truth(str(p))


def test_crop_resize_clips_to_frame():
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    frame[:, :, 1] = 200
    out = crop_resize(frame, (100, 80, 50, 50), (64, 64))  # box overhangs the right/bottom edge
    assert out.shape == (64, 64, 3) and out.dtype == np.uint8
    assert out[:, :, 1].mean() > 190


def test_crop_resize_empty_raises():
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        crop_resize(frame, (130, 130, 10, 10), (64, 64))


def test_detect_face_box_none_on_noise():
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, size=(120, 160, 3), dtype=np.uint8) for _ in range(3)]
    assert detect_face_box(frames, get_face_detector()) is None


def test_process_subject_streams_video_and_skips_faceless(tmp_path):
    """End-to-end on a tiny synthetic AVI: video decodes, no face -> returns None cleanly."""
    subj = tmp_path / "subject99"
    subj.mkdir()
    n = 12
    writer = cv2.VideoWriter(str(subj / "vid.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (160, 120))
    assert writer.isOpened()
    rng = np.random.default_rng(0)
    for _ in range(n):
        writer.write(rng.integers(0, 255, size=(120, 160, 3), dtype=np.uint8))
    writer.release()
    (subj / "ground_truth.txt").write_text(
        " ".join(["0.0"] * n) + "\n" + " ".join(["70"] * n) + "\n" + " ".join(str(i / 30) for i in range(n)) + "\n",
        encoding="utf-8",
    )
    assert process_subject("subject99", dataset_root=str(tmp_path), n_detect_frames=5) is None


def test_process_subject_missing_files(tmp_path):
    assert process_subject("nope", dataset_root=str(tmp_path)) is None

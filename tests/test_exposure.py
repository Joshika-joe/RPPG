"""lock_exposure must never leave the camera darker than it found it."""

import cv2
import numpy as np

from src.live import AUTO_EXPOSURE_OFF, AUTO_EXPOSURE_ON, exposure_candidates, lock_exposure


class FakeCapture:
    """
    Minimal cv2.VideoCapture stand-in. `response` maps an exposure value to the mean
    brightness the sensor would produce; `auto_brightness` is what auto-exposure gives.
    """

    def __init__(self, response, auto_brightness=120.0, start_exposure=-6.0, allow_manual=True):
        self.response = response
        self.auto_brightness = auto_brightness
        self.exposure = start_exposure
        self.allow_manual = allow_manual
        self.auto = AUTO_EXPOSURE_ON
        self.set_calls = []

    def read(self):
        b = self.auto_brightness if self.auto == AUTO_EXPOSURE_ON else self.response(self.exposure)
        return True, np.full((8, 8, 3), np.clip(b, 0, 255), dtype=np.uint8)

    def get(self, prop):
        return self.exposure if prop == cv2.CAP_PROP_EXPOSURE else 0.0

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        if prop == cv2.CAP_PROP_AUTO_EXPOSURE:
            if value == AUTO_EXPOSURE_OFF and not self.allow_manual:
                return False
            self.auto = value
            return True
        if prop == cv2.CAP_PROP_EXPOSURE:
            self.exposure = value
            return True
        return True


def test_locks_when_manual_matches_auto_brightness():
    cap = FakeCapture(response=lambda e: 120.0)  # manual reproduces auto exactly
    ok, msg = lock_exposure(cap, warmup=0.0)
    assert ok, msg
    assert cap.auto == AUTO_EXPOSURE_OFF
    assert "locked" in msg


def test_reverts_to_auto_when_manual_goes_black():
    """The failure seen in the wild: switching to manual lands on a near-black frame."""
    cap = FakeCapture(response=lambda e: 3.0)
    ok, msg = lock_exposure(cap, warmup=0.0)
    assert not ok
    assert cap.auto == AUTO_EXPOSURE_ON  # back on auto, image usable again
    assert "reverted to auto" in msg


def test_steps_exposure_up_until_bright_enough():
    """Driver's own value is too dark, but a longer exposure matches auto."""
    start = -8.0
    cap = FakeCapture(response=lambda e: 120.0 if e >= start + 2 else 10.0, start_exposure=start)
    ok, msg = lock_exposure(cap, warmup=0.0)
    assert ok, msg
    assert cap.exposure >= start + 2
    assert cap.auto == AUTO_EXPOSURE_OFF


def test_gives_up_when_driver_refuses_manual():
    cap = FakeCapture(response=lambda e: 120.0, allow_manual=False)
    ok, msg = lock_exposure(cap, warmup=0.0)
    assert not ok and "refused" in msg


def test_exposure_candidates_scale_by_backend():
    log2 = exposure_candidates(-6.0)  # DirectShow style
    assert log2[0] == -6.0 and max(log2) > -6.0  # tries longer exposures
    absolute = exposure_candidates(100.0)  # V4L2 style
    assert absolute[0] == 100.0 and max(absolute) > 100.0
    assert exposure_candidates(0.0) == []
    assert exposure_candidates(float("nan")) == []

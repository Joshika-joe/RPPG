"""
Live heart-rate (and experimental breathing-rate) estimation from a webcam.

    python -m src.live                              # default camera, results/v3_pearson
    python -m src.live --camera 1 --threads 4
    python -m src.live --video DATASET_2/subject3/vid.avi --no-display   # file mode (testing)

How it works
    * Haar face detection every second; the box is kept fixed while the face stays put
      (the model was trained on one fixed box per recording) and only re-snaps on real moves.
    * Face crops are buffered WITH TIMESTAMPS. Each model call resamples the last 5.0 seconds
      onto exactly 150 frames (linear interpolation in time), which is what the model was
      trained on, so the estimate no longer depends on the camera's frame rate. Feeding 150
      consecutive captured frames instead costs ~2 bpm at 20 fps and ~14 bpm at 15 fps.
    * Window predictions are overlap-added (Hann-weighted, like src.inference) onto a fixed
      30 Hz timeline; HR is the band-limited FFT peak of the last --hr-seconds of it, median
      of recent estimates, and estimates below --min-quality dB are ignored.
    * Breathing rate (experimental). Default `--resp-method chest`: vertical motion of the
      region below the face (phase correlation between consecutive frames), integrated,
      band-passed to 0.1-0.5 Hz, FFT peak over the last --resp-seconds. Independent of the
      pulse model. `--resp-method rsa` instead uses respiratory sinus arrhythmia of the
      predicted BVP (beat-to-beat HR modulation); on UBFC it agrees with the reference PPG's
      own RSA on only ~2 of 7 subjects, so it is kept for comparison only.
      Either way: sit still, needs ~30 s, and treat the number as indicative.

Getting a good signal: bright, steady, front-on light (a window or lamp facing you, not
behind you), no talking or head movement, and a camera running at 25-30 fps. Dim light makes
webcams drop to 15-20 fps and raises sensor noise. Auto-white-balance and auto-exposure fight
rPPG directly - the camera "corrects" the very brightness changes being measured - so this
script turns auto-WB off where the driver allows it (`--auto-wb` to keep it, and
`--manual-exposure` to also lock exposure, which helps but can mis-expose the image).

Keys: q / ESC quit.
"""

import argparse
import collections
import json
import os
import sys
import threading
import time
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np
import torch

from . import config
from .dataset import normalize_clip
from .inference import overlap_add
from .metrics import bandpass_filter, estimate_hr_fft, snr_db
from .models import build_model, load_weights
from .preprocessing import crop_resize, get_face_detector

Box = Tuple[int, int, int, int]

WINDOW_SECONDS = 5.0  # the model's training window
GRID_FPS = config.WINDOW_FRAMES / WINDOW_SECONDS  # 30 Hz timeline predictions live on


# ==============================================================================
# SIGNAL HELPERS
# ==============================================================================
def find_peaks(sig: np.ndarray, fs: float, max_bpm: float = 200.0) -> np.ndarray:
    """Local maxima above 0.3 std with a refractory period of 60/max_bpm seconds."""
    if len(sig) < 3:
        return np.array([], dtype=int)
    thr = 0.3 * sig.std()
    cand = np.flatnonzero((sig[1:-1] > sig[:-2]) & (sig[1:-1] >= sig[2:]) & (sig[1:-1] > thr)) + 1
    min_dist = int(fs * 60.0 / max_bpm)
    keep: List[int] = []
    for i in cand[np.argsort(sig[cand])[::-1]]:  # strongest first
        if all(abs(i - k) >= min_dist for k in keep):
            keep.append(int(i))
    return np.array(sorted(keep), dtype=int)


def breathing_rate_from_bvp(sig: np.ndarray, fs: float, band=(0.1, 0.5)) -> float:
    """
    Respiratory sinus arrhythmia: HR rises on inhale and falls on exhale. Returns breaths/min
    or NaN when the beat series is too short/irregular. `sig` should be band-passed BVP.
    """
    peaks = find_peaks(sig, fs)
    if len(peaks) < 10:
        return float("nan")
    t = peaks / fs
    ibi = np.diff(t)
    ok = (ibi > 0.3) & (ibi < 1.5)
    if ok.sum() < 8:
        return float("nan")
    t_mid = ((t[1:] + t[:-1]) / 2)[ok]
    inst_hr = 60.0 / ibi[ok]
    span = t_mid[-1] - t_mid[0]
    if span < 20.0:
        return float("nan")
    fs_u = 4.0
    t_u = np.arange(t_mid[0], t_mid[-1], 1.0 / fs_u)
    hr_u = np.interp(t_u, t_mid, inst_hr)
    hr_u = (hr_u - hr_u.mean()) * np.hanning(len(hr_u))
    n = len(hr_u) * 8
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_u)
    power = np.abs(np.fft.rfft(hr_u, n=n)) ** 2
    m = (freqs >= band[0]) & (freqs <= band[1])
    if not m.any():
        return float("nan")
    return float(freqs[m][np.argmax(power[m])] * 60.0)


def resample_clip(frames: List[np.ndarray], times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    """
    Linear interpolation in time between buffered uint8 ROI frames.
    Returns (len(target_times), H, W, C) float32 in [0, 1].
    """
    idx = np.clip(np.searchsorted(times, target_times), 1, len(times) - 1)
    t0, t1 = times[idx - 1], times[idx]
    w = ((target_times - t0) / np.maximum(t1 - t0, 1e-9)).astype(np.float32)[:, None, None, None]
    a = np.stack([frames[i - 1] for i in idx]).astype(np.float32)
    b = np.stack([frames[i] for i in idx]).astype(np.float32)
    return (a * (1.0 - w) + b * w) / 255.0


# ==============================================================================
# FACE TRACKING
# ==============================================================================
class FaceTracker:
    """Detects the largest face periodically; keeps the box fixed unless it clearly moves."""

    def __init__(self, every: int = 30, move_frac: float = 0.15):
        self.cascade = get_face_detector()
        self.every = every
        self.move_frac = move_frac
        self.box: Optional[Box] = None
        self.frames_since = every
        self.misses = 0

    def update(self, frame: np.ndarray) -> Optional[Box]:
        self.frames_since += 1
        if self.frames_since < self.every:
            return self.box
        self.frames_since = 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=config.FACE_SCALE_FACTOR, minNeighbors=config.FACE_MIN_NEIGHBORS,
            minSize=config.FACE_MIN_SIZE,
        )
        if len(faces) == 0:
            self.misses += 1
            if self.misses >= 3:
                self.box = None
            return self.box
        self.misses = 0
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        new = (int(x), int(y), int(w), int(h))
        if self.box is None:
            self.box = new
            return self.box
        bx, by, bw, bh = self.box
        moved = (abs((x + w / 2) - (bx + bw / 2)) > self.move_frac * bw
                 or abs((y + h / 2) - (by + bh / 2)) > self.move_frac * bh)
        resized = abs(w - bw) > self.move_frac * bw
        if moved or resized:
            self.box = new
        return self.box


class ChestMotion:
    """
    Tracks vertical motion of the area below the face box with phase correlation and turns
    the integrated displacement into a breathing-rate estimate (breaths / min).
    """

    def __init__(self, seconds: float = 45.0, band=(0.1, 0.5), width: int = 160):
        self.seconds = seconds
        self.band = band
        self.width = width
        self.prev: Optional[np.ndarray] = None
        self.disp: Deque[float] = collections.deque()
        self.times: Deque[float] = collections.deque()
        self.pos = 0.0
        self.trace = np.zeros(0, dtype=np.float32)

    @staticmethod
    def roi_below_face(frame: np.ndarray, box: Box) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x, y, bw, bh = box
        x0, x1 = max(0, int(x - 0.5 * bw)), min(w, int(x + 1.5 * bw))
        y0, y1 = min(h, y + bh), min(h, int(y + bh + 1.2 * bh))
        if y1 - y0 < 20 or x1 - x0 < 20:
            return None
        return frame[y0:y1, x0:x1]

    def update(self, frame: np.ndarray, box: Optional[Box], t: float) -> None:
        roi = None if box is None else self.roi_below_face(frame, box)
        if roi is None:
            self.prev = None
            return
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        scale = self.width / g.shape[1]
        g = cv2.resize(g, (self.width, max(8, int(g.shape[0] * scale)))).astype(np.float32)
        g = (g - g.mean()) / (g.std() + 1e-6)
        if self.prev is not None and self.prev.shape == g.shape:
            (dx, dy), response = cv2.phaseCorrelate(self.prev, g)
            if response > 0.05 and abs(dy) < 0.25 * g.shape[0]:
                self.pos += dy / scale  # back to full-resolution pixels
        self.prev = g
        self.disp.append(self.pos)
        self.times.append(t)
        while self.times and t - self.times[0] > self.seconds:
            self.times.popleft()
            self.disp.popleft()

    def breathing_rate(self) -> float:
        if len(self.times) < 30:
            return float("nan")
        times = np.array(self.times)
        span = times[-1] - times[0]
        if span < 20.0:
            return float("nan")
        fps = (len(times) - 1) / span
        d = np.array(self.disp, dtype=np.float64)
        d = d - np.polyval(np.polyfit(np.arange(len(d)), d, 1), np.arange(len(d)))  # detrend
        filt = bandpass_filter(d, fps, band=self.band, transition_hz=0.03)
        self.trace = filt.astype(np.float32)
        w = filt * np.hanning(len(filt))
        n = len(w) * 8
        freqs = np.fft.rfftfreq(n, d=1.0 / fps)
        power = np.abs(np.fft.rfft(w, n=n)) ** 2
        m = (freqs >= self.band[0]) & (freqs <= self.band[1])
        if not m.any() or power[m].max() <= 0:
            return float("nan")
        return float(freqs[m][np.argmax(power[m])] * 60.0)


# ==============================================================================
# ESTIMATOR
# ==============================================================================
class LiveEstimator:
    def __init__(self, model, normalize: str, device, hr_seconds: float, resp_seconds: float,
                 resp_method: str = "chest", min_quality: float = -3.0, smooth: int = 7):
        self.model = model
        self.normalize = normalize
        self.device = device
        self.window = config.WINDOW_FRAMES
        self.hr_seconds = hr_seconds
        self.resp_seconds = resp_seconds
        self.resp_method = resp_method
        self.min_quality = min_quality
        self.chest = ChestMotion(seconds=resp_seconds)

        # capture buffer: enough for WINDOW_SECONDS even at 60 fps, plus margin
        maxlen = int(60 * (WINDOW_SECONDS + 1))
        self.frames: Deque[np.ndarray] = collections.deque(maxlen=maxlen)
        self.times: Deque[float] = collections.deque(maxlen=maxlen)
        self.t0: Optional[float] = None
        self.windows: Deque[Tuple[int, np.ndarray]] = collections.deque()
        self.max_windows = 128
        self.lock = threading.Lock()

        self.hr = float("nan")
        self.hr_smooth: Deque[float] = collections.deque(maxlen=smooth)
        self.resp = float("nan")
        self.resp_smooth: Deque[float] = collections.deque(maxlen=5)
        self.quality_db = float("nan")
        self.trace = np.zeros(0, dtype=np.float32)
        self.capture_fps = float("nan")
        self.buffered_seconds = 0.0

    # -- capture side ------------------------------------------------------
    def push(self, roi: np.ndarray, t: float) -> None:
        with self.lock:
            if self.t0 is None:
                self.t0 = t
            self.frames.append(roi)
            self.times.append(t)
            if len(self.times) > 1:
                span = self.times[-1] - self.times[0]
                self.buffered_seconds = span
                if span > 0:
                    self.capture_fps = (len(self.times) - 1) / span

    def ready(self) -> bool:
        return self.buffered_seconds >= WINDOW_SECONDS

    def snapshot(self) -> Optional[Tuple[np.ndarray, int]]:
        """Resamples the last WINDOW_SECONDS onto 150 frames; returns (clip, grid_start)."""
        with self.lock:
            if self.t0 is None or len(self.times) < 4:
                return None
            times = np.array(self.times)
            frames = list(self.frames)
            t0 = self.t0
        t_end = times[-1]
        t_start = t_end - WINDOW_SECONDS
        if t_start < times[0]:
            return None
        target = np.linspace(t_start, t_end, self.window)
        return resample_clip(frames, times, target), int(round((t_start - t0) * GRID_FPS))

    # -- inference side ----------------------------------------------------
    def infer_window(self, clip: np.ndarray, grid_start: int) -> None:
        x = np.ascontiguousarray(np.transpose(clip, (3, 0, 1, 2)))  # (C, T, H, W)
        mean_frame = torch.from_numpy(np.ascontiguousarray(x.mean(axis=1)))[None]
        x = normalize_clip(x, self.normalize).astype(np.float32)
        with torch.no_grad():
            pred = self.model(torch.from_numpy(x)[None].to(self.device), appearance=mean_frame.to(self.device))
        with self.lock:
            self.windows.append((grid_start, pred[0].cpu().numpy()))
            while len(self.windows) > self.max_windows:
                self.windows.popleft()
        self.update_estimates()

    def update_estimates(self) -> None:
        with self.lock:
            windows = list(self.windows)
        if not windows:
            return
        offset = windows[0][0]
        n = windows[-1][0] + self.window - offset
        cont = overlap_add([(s - offset, p) for s, p in windows], n, self.window)
        if len(cont) < self.window:
            return
        filt = bandpass_filter(cont, GRID_FPS)

        n_hr = min(len(filt), int(self.hr_seconds * GRID_FPS))
        seg = filt[-n_hr:]
        hr = estimate_hr_fft(seg, GRID_FPS)
        self.quality_db = snr_db(seg, hr, GRID_FPS)
        self.trace = seg.astype(np.float32)
        if np.isfinite(hr) and (not np.isfinite(self.quality_db) or self.quality_db >= self.min_quality):
            self.hr_smooth.append(hr)
        if self.hr_smooth:
            self.hr = float(np.median(self.hr_smooth))

        if self.resp_method == "rsa":
            n_resp = int(self.resp_seconds * GRID_FPS)
            resp = breathing_rate_from_bvp(filt[-n_resp:], GRID_FPS) if len(filt) >= n_resp else float("nan")
        else:
            resp = self.chest.breathing_rate()
        if np.isfinite(resp):
            self.resp_smooth.append(resp)
            self.resp = float(np.median(self.resp_smooth))

    @property
    def signal_ok(self) -> bool:
        return np.isfinite(self.quality_db) and self.quality_db >= self.min_quality


# ==============================================================================
# DISPLAY
# ==============================================================================
def draw_overlay(frame: np.ndarray, est: LiveEstimator, box: Optional[Box]) -> np.ndarray:
    h, w = frame.shape[:2]
    if box is not None:
        x, y, bw, bh = box
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 200, 0), 2)

    strip_h = 92
    canvas = np.zeros((h + strip_h, w, 3), dtype=np.uint8)
    canvas[:h] = frame
    tr = est.trace
    if len(tr) > 2:
        tr = (tr - tr.min()) / (tr.max() - tr.min() + 1e-6)
        xs = np.linspace(0, w - 1, len(tr)).astype(int)
        ys = (h + strip_h - 8 - tr * (strip_h - 16)).astype(int)
        cv2.polylines(canvas, [np.stack([xs, ys], axis=1)], False, (0, 220, 255), 1)

    def put(text, y, color=(255, 255, 255), scale=0.7):
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    if box is None:
        put("No face detected", 30, (0, 0, 255))
    elif not est.ready():
        put(f"Buffering face... {est.buffered_seconds:.1f} / {WINDOW_SECONDS:.0f} s", 30, (0, 200, 255))
    else:
        ok = est.signal_ok
        put(f"HR  {est.hr:5.1f} bpm" if np.isfinite(est.hr) else "HR  --", 30,
            (0, 255, 0) if ok else (0, 200, 255), 0.9)
        method = "chest motion" if est.resp_method == "chest" else "pulse RSA"
        put(f"Breathing  {est.resp:4.1f} /min ({method}, experimental)" if np.isfinite(est.resp)
            else f"Breathing  -- ({method}; needs ~30 s, sit still)", 58, (200, 200, 200), 0.6)
        q = f"quality {est.quality_db:+.1f} dB" if np.isfinite(est.quality_db) else "quality --"
        warn = "" if ok else "  weak signal - more light, hold still"
        fps_warn = "  (low fps: more light)" if np.isfinite(est.capture_fps) and est.capture_fps < 24 else ""
        put(f"{q}   {est.capture_fps:.1f} fps{fps_warn}{warn}", 84, (200, 200, 200), 0.55)
    return canvas


# ==============================================================================
# CAPTURE
# ==============================================================================
def open_capture(args):
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit(f"Could not open video {args.video}")
        return cap

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. Try --camera 1.")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # raw YUY2 is often capped at 5-10 fps
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not args.auto_wb:
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    if args.manual_exposure:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # DSHOW: 0.25 = manual, 0.75 = auto
    print(
        f"Camera {args.camera}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
        f" @ {cap.get(cv2.CAP_PROP_FPS):.0f} fps requested, auto-WB "
        f"{'on' if args.auto_wb else 'off'}, exposure {'manual' if args.manual_exposure else 'auto'}"
    )
    return cap


def load_run(run: str, device):
    run_dir = run if os.path.isabs(run) else os.path.join(config.PROJECT_ROOT, run)
    with open(os.path.join(run_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model = build_model(cfg["model"])
    load_weights(model, os.path.join(run_dir, "best.pth"), device)
    return model, cfg["normalize"]


# ==============================================================================
# MAIN LOOP
# ==============================================================================
def main() -> None:
    p = argparse.ArgumentParser(description="Live rPPG heart-rate estimation")
    p.add_argument("--run", default="results/v3_pearson")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--video", help="read a video file instead of the camera (uses its fps)")
    p.add_argument("--update-every", type=float, default=0.5, help="seconds between model updates")
    p.add_argument("--hr-seconds", type=float, default=10.0)
    p.add_argument("--resp-seconds", type=float, default=45.0)
    p.add_argument("--resp-method", default="chest", choices=["chest", "rsa"])
    p.add_argument("--min-quality", type=float, default=-3.0, help="ignore HR estimates below this SNR (dB)")
    p.add_argument("--smooth", type=int, default=7, help="median filter length over recent HR estimates")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--auto-wb", action="store_true", help="leave auto white balance on")
    p.add_argument("--manual-exposure", action="store_true", help="lock exposure (helps rPPG, may mis-expose)")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--log-every", type=float, default=2.0, help="seconds between stdout lines")
    p.add_argument("--simulate-fps", type=float, default=0.0, help="drop frames to emulate a slower camera (testing)")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    device = config.get_device()
    model, normalize = load_run(args.run, device)
    print(f"Model from {args.run} on {device}, normalize={normalize}")

    cap = open_capture(args)
    video_fps = cap.get(cv2.CAP_PROP_FPS) if args.video else None
    tracker = FaceTracker()
    est = LiveEstimator(model, normalize, device, args.hr_seconds, args.resp_seconds,
                        resp_method=args.resp_method, min_quality=args.min_quality, smooth=args.smooth)
    worker: Optional[threading.Thread] = None
    frame_idx = 0
    last_kept: Optional[float] = None
    last_infer = -1e9
    last_log = -1e9
    print("Running. Press q / ESC in the window to quit." if not args.no_display else "Running headless.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = frame_idx / video_fps if args.video else time.perf_counter()
            frame_idx += 1
            if args.simulate_fps > 0:
                if last_kept is not None and t - last_kept < 1.0 / args.simulate_fps - 1e-9:
                    continue
                last_kept = t

            box = tracker.update(frame)
            if args.resp_method == "chest":
                est.chest.update(frame, box, t)
            if box is not None:
                try:
                    est.push(crop_resize(frame, box, config.TARGET_SIZE), t)
                except ValueError:
                    pass

            if est.ready() and t - last_infer >= args.update_every:
                snap = est.snapshot()
                if snap is not None:
                    last_infer = t
                    if args.video:
                        est.infer_window(*snap)  # files are read faster than real time
                    elif worker is None or not worker.is_alive():
                        worker = threading.Thread(target=est.infer_window, args=snap, daemon=True)
                        worker.start()

            if t - last_log >= args.log_every and est.ready():
                last_log = t
                print(
                    f"t={t:8.1f}s  HR {est.hr:6.1f} bpm  breathing {est.resp:5.1f}/min  "
                    f"quality {est.quality_db:+5.1f} dB  fps {est.capture_fps:4.1f}"
                    f"{'' if est.signal_ok else '  (weak)'}",
                    flush=True,
                )

            if not args.no_display:
                shown = frame if args.video else cv2.flip(frame, 1)
                shown_box = box
                if box is not None and not args.video:  # mirror the box with the image
                    x, y, bw, bh = box
                    shown_box = (frame.shape[1] - x - bw, y, bw, bh)
                cv2.imshow("rPPG live", draw_overlay(shown, est, shown_box))
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

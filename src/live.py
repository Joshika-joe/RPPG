"""
Live heart-rate (and experimental breathing-rate) estimation from a webcam.

    python -m src.live                              # default camera, results/v3_pearson
    python -m src.live --camera 1 --threads 4
    python -m src.live --video DATASET_2/subject3/vid.avi --no-display   # file mode (testing)

How it works
    * Haar face detection every second; the box is kept fixed while the face stays put
      (the model was trained on one fixed box per recording) and only re-snaps on real moves.
    * A rolling 5-s buffer of 64x64 face crops. Every --update-every frames the newest
      window is temporally normalized and run through the model in a background thread.
    * Window predictions are overlap-added (Hann-weighted, like src.inference) into one
      continuous BVP; HR is the band-limited FFT peak of the last --hr-seconds of it.
    * Breathing rate (experimental). Default `--resp-method chest`: vertical motion of the
      region below the face (phase correlation between consecutive frames), integrated,
      band-passed to 0.1-0.5 Hz, FFT peak over the last --resp-seconds. Independent of the
      pulse model. `--resp-method rsa` instead uses respiratory sinus arrhythmia of the
      predicted BVP (beat-to-beat HR modulation); on UBFC it agrees with the reference PPG's
      own RSA on only ~2 of 7 subjects, so it is kept for comparison only.
      Either way: sit still, needs ~30 s, and treat the number as indicative.

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
        moved = abs((x + w / 2) - (bx + bw / 2)) > self.move_frac * bw or abs((y + h / 2) - (by + bh / 2)) > self.move_frac * bh
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
        self.prev_shape = None
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
    def __init__(self, model, normalize: str, device, window: int, hr_seconds: float, resp_seconds: float,
                 resp_method: str = "chest"):
        self.model = model
        self.resp_method = resp_method
        self.chest = ChestMotion(seconds=resp_seconds)
        self.normalize = normalize
        self.device = device
        self.window = window
        self.hr_seconds = hr_seconds
        self.resp_seconds = resp_seconds
        self.history_frames = int(resp_seconds * 35) + window  # generous ring for the highest fps

        self.frames: Deque[np.ndarray] = collections.deque(maxlen=window)
        self.times: Deque[float] = collections.deque(maxlen=self.history_frames)
        self.n_seen = 0  # global frame counter
        self.windows: Deque[Tuple[int, np.ndarray]] = collections.deque()
        self.lock = threading.Lock()

        self.hr = float("nan")
        self.hr_smooth: Deque[float] = collections.deque(maxlen=5)
        self.resp = float("nan")
        self.resp_smooth: Deque[float] = collections.deque(maxlen=5)
        self.quality_db = float("nan")
        self.trace = np.zeros(0, dtype=np.float32)
        self.fps = float("nan")
        self.busy = False

    # -- capture side ------------------------------------------------------
    def push(self, roi: np.ndarray, t: float) -> None:
        with self.lock:
            self.frames.append(roi)
            self.times.append(t)
            self.n_seen += 1

    def ready(self) -> bool:
        return len(self.frames) == self.window

    def snapshot(self):
        with self.lock:
            clip = np.stack(self.frames)
            return clip, self.n_seen - self.window

    # -- inference side ----------------------------------------------------
    def infer_window(self, clip: np.ndarray, start: int) -> None:
        x = clip.astype(np.float32) / 255.0  # (T, H, W, C)
        x = np.ascontiguousarray(np.transpose(x, (3, 0, 1, 2)))  # (C, T, H, W)
        mean_frame = torch.from_numpy(np.ascontiguousarray(x.mean(axis=1)))[None]
        x = normalize_clip(x, self.normalize).astype(np.float32)
        with torch.no_grad():
            pred = self.model(torch.from_numpy(x)[None].to(self.device), appearance=mean_frame.to(self.device))
        pred = pred[0].cpu().numpy()
        with self.lock:
            self.windows.append((start, pred))
            while self.windows and self.windows[0][0] < self.n_seen - self.history_frames:
                self.windows.popleft()
        self.update_estimates()

    def update_estimates(self) -> None:
        with self.lock:
            times = np.array(self.times)
            n_seen = self.n_seen
            windows = list(self.windows)
        if len(times) < self.window or not windows:
            return
        fps = (len(times) - 1) / max(times[-1] - times[0], 1e-6)
        self.fps = fps

        # continuous signal over the retained history
        n_hist = min(self.history_frames, n_seen)
        offset = n_seen - n_hist
        local = [(s - offset, p) for s, p in windows if s - offset >= 0]
        if not local:
            return
        cont = overlap_add(local, n_hist, self.window)
        first = local[0][0]
        cont = cont[first:]  # drop the uncovered head
        if len(cont) < self.window:
            return
        filt = bandpass_filter(cont, fps)

        n_hr = min(len(filt), int(self.hr_seconds * fps))
        seg = filt[-n_hr:]
        hr = estimate_hr_fft(seg, fps)
        q = snr_db(seg, hr, fps)
        self.hr_smooth.append(hr)
        self.hr = float(np.median(self.hr_smooth))
        self.quality_db = q
        self.trace = seg.astype(np.float32)

        if self.resp_method == "rsa":
            n_resp = int(self.resp_seconds * fps)
            resp = breathing_rate_from_bvp(filt[-n_resp:], fps) if len(filt) >= n_resp else float("nan")
        else:
            resp = self.chest.breathing_rate()
        if np.isfinite(resp):
            self.resp_smooth.append(resp)
            self.resp = float(np.median(self.resp_smooth))


# ==============================================================================
# DISPLAY
# ==============================================================================
def draw_overlay(frame: np.ndarray, est: LiveEstimator, box: Optional[Box], seconds_buffered: float) -> np.ndarray:
    h, w = frame.shape[:2]
    if box is not None:
        x, y, bw, bh = box
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 200, 0), 2)

    strip_h = 90
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
        put(f"Buffering face... {seconds_buffered:.1f} s", 30, (0, 200, 255))
    else:
        good = np.isfinite(est.quality_db) and est.quality_db > -5.0
        hr_txt = f"HR  {est.hr:5.1f} bpm" if np.isfinite(est.hr) else "HR  --"
        put(hr_txt, 30, (0, 255, 0) if good else (0, 200, 255), 0.9)
        method = "chest motion" if est.resp_method == "chest" else "pulse RSA"
        resp_txt = (f"Breathing  {est.resp:4.1f} /min ({method}, experimental)" if np.isfinite(est.resp)
                    else f"Breathing  -- ({method}; needs ~30 s, sit still)")
        put(resp_txt, 58, (200, 200, 200), 0.6)
        q = f"quality {est.quality_db:+.1f} dB" if np.isfinite(est.quality_db) else "quality --"
        put(f"{q}   {est.fps:.1f} fps   {'hold still' if not good else ''}", 82, (200, 200, 200), 0.55)
    return canvas


# ==============================================================================
# MAIN LOOP
# ==============================================================================
def load_run(run: str, device):
    run_dir = run if os.path.isabs(run) else os.path.join(config.PROJECT_ROOT, run)
    with open(os.path.join(run_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model = build_model(cfg["model"])
    load_weights(model, os.path.join(run_dir, "best.pth"), device)
    return model, cfg["normalize"]


def open_capture(args):
    if args.video:
        cap = cv2.VideoCapture(args.video)
    elif sys.platform == "win32":
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    else:
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {'video ' + args.video if args.video else 'camera ' + str(args.camera)}")
    return cap


def main() -> None:
    p = argparse.ArgumentParser(description="Live rPPG heart-rate estimation")
    p.add_argument("--run", default="results/v3_pearson")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--video", help="read a video file instead of the camera (uses its fps)")
    p.add_argument("--update-every", type=int, default=15, help="frames between model updates")
    p.add_argument("--hr-seconds", type=float, default=10.0)
    p.add_argument("--resp-seconds", type=float, default=45.0)
    p.add_argument("--resp-method", default="chest", choices=["chest", "rsa"])
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--log-every", type=float, default=2.0, help="seconds between stdout lines")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    device = config.get_device()
    model, normalize = load_run(args.run, device)
    print(f"Model from {args.run} on {device}, normalize={normalize}")

    cap = open_capture(args)
    video_fps = cap.get(cv2.CAP_PROP_FPS) if args.video else None
    tracker = FaceTracker()
    est = LiveEstimator(model, normalize, device, config.WINDOW_FRAMES, args.hr_seconds, args.resp_seconds,
                        resp_method=args.resp_method)
    worker: Optional[threading.Thread] = None
    frame_idx = 0
    last_log = -1e9
    print("Running. Press q / ESC in the window to quit." if not args.no_display else "Running headless.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = frame_idx / video_fps if args.video else time.perf_counter()
            frame_idx += 1

            box = tracker.update(frame)
            if args.resp_method == "chest":
                est.chest.update(frame, box, t)
            if box is not None:
                try:
                    est.push(crop_resize(frame, box, config.TARGET_SIZE), t)
                except ValueError:
                    pass

            if est.ready() and frame_idx % args.update_every == 0:
                clip, start = est.snapshot()
                if args.video:
                    est.infer_window(clip, start)  # synchronous: files are read faster than real time
                elif worker is None or not worker.is_alive():
                    worker = threading.Thread(target=est.infer_window, args=(clip, start), daemon=True)
                    worker.start()

            if t - last_log >= args.log_every and est.ready():
                last_log = t
                print(
                    f"t={t:6.1f}s  HR {est.hr:6.1f} bpm  breathing {est.resp:5.1f}/min  "
                    f"quality {est.quality_db:+5.1f} dB  fps {est.fps:4.1f}",
                    flush=True,
                )

            if not args.no_display:
                shown = cv2.flip(frame, 1) if not args.video else frame
                shown_box = box
                if box is not None and not args.video:  # mirror the box with the image
                    x, y, bw, bh = box
                    shown_box = (frame.shape[1] - x - bw, y, bw, bh)
                buffered = len(est.frames) / (est.fps if np.isfinite(est.fps) and est.fps > 0 else 30.0)
                cv2.imshow("rPPG live", draw_overlay(shown, est, shown_box, buffered))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

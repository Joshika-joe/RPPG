# rPPG — contactless pulse estimation from face video (UBFC-rPPG)

Deep-learning pipeline that estimates the Blood Volume Pulse (BVP) waveform — and from it heart
rate — from a face video. Trained and evaluated on UBFC-rPPG (42 subjects, subject-level
train / val / test split of 29 / 6 / 7).

**Best model:** `v3` (2D+1D CNN with learned spatial attention, 168 K parameters), trained with a
Pearson-correlation loss. On the held-out test subjects, continuous inference gives
**HR MAE 0.37 bpm, RMSE 0.54 bpm, 100 % of 10-s segments within 5 bpm, waveform r = 0.81**.
Training takes ~1 h on a laptop CPU. Weights: `results/v3_pearson/best.pth`.

## Results

Held-out **test** split (7 subjects, ~60 s each). HR reference = FFT peak of the ground-truth
BVP with harmonic-aware peak picking; per-subject frame rates are used throughout.

### Continuous inference (overlap-add, 10-s segments — the standard UBFC protocol)

| run | model | data pipeline | loss | Pearson r | HR MAE | HR RMSE | within 5 bpm | SNR | whole-recording r |
|---|---|---|---|---|---|---|---|---|---|
| legacy v2 | 3D-CNN 507 K | 362 windows, raw pixels | MSE | 0.648 | 2.39 bpm | 9.42 | 92.9 % | 1.30 dB | 0.642 |
| v2_newpipe | 3D-CNN 507 K | stride-30 windows, temporal norm, aug. | MSE | 0.759 | 0.68 | 1.99 | 97.6 % | 2.21 | 0.750 |
| v3_newpipe | 2D+1D 168 K | same | MSE | 0.762 | 0.55 | 1.64 | 97.6 % | 2.76 | 0.754 |
| **v3_pearson** | 2D+1D 168 K | same | (1−r) + 0.2·MSE | **0.813** | **0.37 bpm** | **0.54** | **100 %** | **3.72 dB** | **0.801** |

Each row changes exactly one thing relative to the row above (data → architecture → loss/optimizer),
so the deltas are attributable.

### Isolated 5-s windows (stride 150, no post-processing)

| run | MSE | Pearson r | HR MAE | HR RMSE | within 5 bpm | SNR | amplitude ratio |
|---|---|---|---|---|---|---|---|
| legacy v2 | 0.671 | 0.634 | 2.45 bpm | 6.83 | 92.2 % | 0.73 dB | 0.47 |
| v2_newpipe | 0.431 | 0.759 | 1.22 | 2.48 | 96.7 % | 2.01 | 0.73 |
| v3_newpipe | 0.435 | 0.753 | 1.73 | 5.21 | 95.6 % | 2.32 | 0.77 |
| **v3_pearson** | **0.348** | **0.811** | **1.07 bpm** | **1.62** | **98.9 %** | **3.27 dB** | **0.79** |

Amplitude ratio (prediction std / target std) shows the MSE regression-to-the-mean effect
(0.47) that the Pearson loss removes. On the same v3_pearson checkpoint, overlap-add alone
lowers 5-s HR MAE from 1.07 to 0.80 bpm; 10-s segments bring it to 0.37 bpm.

Validation numbers (used for early stopping, so optimistic): legacy r 0.710 / HR MAE 0.79;
v2_newpipe 0.817 / 0.90; v3_newpipe 0.824 / 0.98; v3_pearson 0.853 / 0.69.
A rerun of the v3_pearson recipe selected on val r instead of HR MAE (`v3_pearson_selr`) ties
on windowed test metrics (r 0.819, HR MAE 1.06) and is slightly worse on continuous
(HR MAE 0.57), so the result is not a lucky epoch and v3_pearson stays the reference.

Remaining errors concentrate in subject12's first 10 s (irregular reference waveform) and
subject48 (lowest-SNR video). `results/v3_pearson_test_continuous_seg300/continuous.png` shows
the first 20 s of every test subject; `results/v3_pearson_test/attention.png` shows the learned
spatial attention landing on forehead and cheeks and avoiding hair, eyes, background and beard,
with no supervision on location.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate                # Windows;  source .venv/bin/activate elsewhere
pip install -e ".[dev]"               # package + pytest
python -m pytest                      # 36 tests, ~4 s, no dataset needed
```

An NVIDIA GPU is picked up automatically; install the CUDA torch build first
(`pip install torch --index-url https://download.pytorch.org/whl/cu124`). On a 12-core laptop
CPU, v3 trains at ~2 min/epoch (`--threads 8` keeps it cooler), v2 at ~12 min/epoch.

The raw dataset (70 GB) and the preprocessing cache are not in git: put UBFC-rPPG at
`DATASET_2/subjectN/{vid.avi, ground_truth.txt}`. Trained runs (`results/<run>/*.pth`,
history, metrics, plots) **are** tracked, so a model trained elsewhere can be pushed back
and evaluated here.

## Pipeline

### 1. Preprocess (once, ~3.5 min for 42 subjects)

```bash
python -m src.preprocessing            # --force to regenerate, --subjects to restrict
```

Per subject: decode `vid.avi` frame-by-frame (never the whole video in RAM), detect the
largest face in the first 30 frames, take the median box, crop + resize every frame to 64x64,
align BVP / HR / timestamps per frame. Output: `checkpoints/subjects/<id>/` with `roi.npy`
(uint8, T x 64 x 64 x 3, BGR), `bvp.npy`, `hr.npy`, `ts.npy`, `meta.json`. Windowing is *not*
done here, so stride, window length, normalization and augmentation are training-time choices.

Data facts: subjects 25, 26, 27 are recorded at ~23.3 fps (others ~29.5); the real fps is in
`meta.json` and used for every HR computation. UBFC's oximeter HR line drops to 1–4 bpm for
subjects 11, 18, 20, so the HR reference is the FFT of the ground-truth BVP and the oximeter
value is reported only as `hr_mae_vs_oximeter_bpm`.

### 2. Train

```bash
# best recipe
python -m src.train --model v3 --run my_run --loss pearson --optimizer adamw --lr 1e-3 --wd 1e-2 \
    --scheduler cosine --warmup-epochs 3 --grad-clip 1.0 --select-on hr_mae --patience 15 --threads 8
python -m src.train --model v3 --run my_run --resume       # continue after an interruption
```

Defaults (`src/config.py`): 150-frame windows, train stride 30 (1,754 overlapping windows from
29 subjects), 400 random windows per epoch, batch 4, temporal input normalization, augmentation
(h-flip, brightness / contrast, ±4 px shift — each drawn once per window so nothing varies
over time), early stopping on the `--select-on` metric. Every option is a CLI flag. Each run
writes `results/<run>/{config.json, history.json, best.pth, last.pth, loss_curve.png}`.

`--loss pearson` is (1 − Pearson r) + 0.2·MSE: scale-invariant, so the model is rewarded for
waveform shape rather than for hedging toward zero amplitude.

### 3. Evaluate

```bash
python -m src.evaluate --run results/v3_pearson --split test                       # isolated windows
python -m src.evaluate --run results/v3_pearson --split test --continuous --stride 30 --segment-frames 300
python -m src.evaluate --checkpoint best_rppg_model_v2.pth --model v2 --normalize raw --split test
```

Writes `metrics.json` (summary + per-subject), `per_window.csv`, and plots (best / median /
worst waveforms, HR scatter, Bland–Altman, attention map for v3, `continuous.png` in
continuous mode). Metrics: MSE, per-window Pearson r, HR MAE / RMSE (FFT peak in 0.7–4 Hz),
fraction within 5 bpm, de Haan SNR (leakage caps a perfect 5-s sine at ~6 dB, so compare
relatively), amplitude ratio, and in continuous mode the whole-recording r.

`--continuous` slides the model over each recording with `--stride`, standardizes each window's
prediction, blends them with a Hann weight into one continuous BVP (`src/inference.py`), and
scores non-overlapping `--segment-frames` segments. `--bandpass` (0.7–4 Hz on the prediction)
changes HR MAE by < 0.05 bpm and is off by default.

## Live webcam demo

```bash
python -m src.live                       # default camera, results/v3_pearson, q / ESC to quit
python -m src.live --camera 1 --threads 4
python -m src.live --video DATASET_2/subject3/vid.avi --no-display   # file mode, prints estimates
```

Shows the face box, the live BVP trace, **heart rate** from the last 10 s, a signal-quality
readout, and an **experimental breathing rate**. Sit still, face the camera in steady light,
and wait ~10 s for the HR and ~45 s for the breathing estimate.

- HR: rolling 5-s windows through v3 every 0.5 s (background thread), overlap-added exactly
  like `--continuous`, band-limited FFT peak, median of the last 5 estimates. In file mode on
  the test subjects it tracks the reference within ~1–3 bpm throughout, including HR changes.
- Breathing (`--resp-method chest`, default): vertical motion of the region below the face
  (phase correlation), integrated, band-passed to 0.1–0.5 Hz. On UBFC videos it finds a sharp
  peak with a clean 2x harmonic per subject (23–27 /min; the subjects were playing a timed
  math game), but UBFC has no respiration labels, so this is **unvalidated** — check it by
  counting your own breaths for 30 s. `--resp-method rsa` derives breathing from
  beat-to-beat HR modulation of the predicted pulse instead; it agrees with the reference
  PPG's own RSA on only 2 of 7 test subjects and is kept for comparison only.

The model was trained on frontal, still, indoor subjects at ~30 fps. Head movement, talking,
backlighting and changing light will degrade it; the quality readout (SNR of the last 10 s)
goes amber when the estimate is doubtful.

## Models

**v2** — 3D-CNN (507 K params): four Conv3d blocks, spatial global average pool, 1D temporal
conv head. `best_rppg_model_v2.pth` is the legacy checkpoint trained on the original pipeline.

**v3** — 2D+1D (168 K params): a shared 2D encoder runs on every frame (64 → 8x8 feature map);
a small appearance branch turns the window's *mean frame* into an 8x8 spatial attention mask
(temporal normalization removes appearance from the frames themselves, so "where is the skin"
has to come from somewhere else); attention-weighted pooling gives a (C, T) sequence; five
dilated residual 1D blocks (receptive field 125 frames) and a 1x1 head produce the BVP. ~4.7x
cheaper per training step than v2 on CPU. Output length follows the input length.

### Input normalization

`temporal` (default): subtract each pixel's mean over the window, then divide by one global
std. This removes the skin-colour DC term and static illumination so the ~1 % pulse component
is what the network sees, while keeping the relative AC amplitude between skin and background
pixels. `raw` (pixels / 255) is what the legacy checkpoint was trained on.

## Layout

```
src/
  config.py         paths, windowing, normalization, augmentation, metric and training defaults
  preprocessing.py  video -> per-subject 64x64 face-ROI sequence (+ BVP / HR / timestamps)
  dataset.py        WindowDataset: on-the-fly windowing, normalization, augmentation
  models.py         model registry (v2: 3D-CNN, v3: 2D+1D with appearance attention)
  metrics.py        Pearson r, HR MAE / RMSE, SNR, band-pass, per-subject tables (numpy only)
  inference.py      continuous overlap-add inference over whole recordings
  train.py          training loop (losses, AdamW / cosine, grad clip, seeded, resumable, history)
  evaluate.py       metrics.json + per-window CSV + plots, windowed or continuous
  live.py           webcam demo: live HR (+ experimental breathing rate)
tests/              pytest suite (synthetic data; no dataset or trained model required)
splits/             train_subjects.txt, val_subjects.txt, test_subjects.txt
checkpoints/subjects/<id>/   preprocessing cache (not tracked)
results/<run>/               config.json, history.json, best.pth, last.pth, loss_curve.png
results/<run>_<split>[_continuous_segN]/   metrics.json, per_window.csv, *.png
notebooks/          original exploration notebooks (superseded by src/)
docs/               background notes
archive/v1/         retired v1 (78.7 M params, never learned: val MSE 0.997 = predicting zeros; not tracked)
DATASET_2/          raw UBFC-rPPG (not tracked)
```

## Collaborator workflow

```bash
git clone https://github.com/Joshika-joe/RPPG.git && cd RPPG
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[dev]" && python -m pytest
# put UBFC-rPPG in DATASET_2/, then:
python -m src.preprocessing
python -m src.train --model v3 --run my_run --loss pearson --optimizer adamw --lr 1e-3 --wd 1e-2 \
    --scheduler cosine --warmup-epochs 3 --grad-clip 1.0 --select-on hr_mae --patience 15
python -m src.evaluate --run results/my_run --split test --continuous --stride 30 --segment-frames 300
git add results/my_run results/my_run_test_continuous && git commit -m "Add my_run" && git push
```

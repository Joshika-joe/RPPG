# rPPG — contactless pulse estimation from face video (UBFC-rPPG)

Deep-learning pipeline that estimates the Blood Volume Pulse (BVP) waveform, and from it
heart rate, from a 5-second face video clip. Trained and evaluated on UBFC-rPPG (42 subjects,
subject-level train / val / test split).

## Layout

```
src/
  config.py         paths, windowing, normalization, augmentation, training defaults
  preprocessing.py  video -> per-subject 64x64 face-ROI sequence (+ BVP / HR / timestamps)
  dataset.py        WindowDataset: on-the-fly windowing, normalization, augmentation
  models.py         model registry (v2: 3D-CNN)
  metrics.py        Pearson r, HR MAE / RMSE, SNR, band-pass, per-subject tables
  train.py          training loop (seeded, resumable, per-epoch val metrics, history)
  evaluate.py       metrics.json + per-window CSV + waveform / HR scatter / Bland-Altman plots
splits/             train_subjects.txt, val_subjects.txt, test_subjects.txt
checkpoints/
  subjects/<id>/    roi.npy (uint8, T x 64 x 64 x 3, BGR), bvp.npy, hr.npy, ts.npy, meta.json
results/<run>/      config.json, history.json, best.pth, last.pth, loss_curve.png
results/<run>_<split>/  metrics.json, per_window.csv, *.png
notebooks/          original exploration notebooks (superseded by src/)
docs/               background notes
archive/v1/         retired v1 model (78.7M params; never learned - val MSE 0.997 = predicting zeros)
DATASET_2/          raw UBFC-rPPG (not tracked)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

All commands run from the project root as modules (`python -m src.<name>`).

## 1. Preprocess (once, ~3.5 min for 42 subjects)

```bash
python -m src.preprocessing
```

Per subject: decode `vid.avi` frame-by-frame, detect the largest face in the first 30 frames,
take the median box, crop + resize every frame to 64x64, align BVP / HR / timestamps per frame.
Windowing is **not** done here, so stride and window length are training-time choices.

Note: subjects 25, 26, 27 are recorded at ~23.3 fps, all others at ~29.5 fps. The real
per-subject fps is stored in `meta.json` and used for every HR computation.

## 2. Train

```bash
python -m src.train --model v2 --run v2_newpipe
python -m src.train --model v2 --run v2_newpipe --resume     # continue after interruption
```

Defaults (`src/config.py`): 150-frame windows, train stride 30 (1,754 overlapping windows
from 29 subjects), 400 random windows per epoch, temporal input normalization, augmentation
(h-flip, brightness / contrast, ±4 px shift — all constant over time), Adam 1e-4, batch 4,
MSE loss, early stopping on val MSE (patience 8). Every option is a CLI flag.

## 3. Evaluate

```bash
python -m src.evaluate --run results/v2_newpipe --split test
python -m src.evaluate --checkpoint best_rppg_model_v2.pth --model v2 --normalize raw --split val
```

Reports MSE, per-window Pearson r, HR MAE / RMSE (FFT peak in 0.7–4 Hz, using the real fps),
fraction within 5 bpm, SNR (de Haan), and the prediction / target amplitude ratio, overall and
per subject. `--bandpass` filters predictions to 0.7–4 Hz first.

The HR reference is the FFT of the ground-truth BVP. UBFC's oximeter HR line (line 2 of
`ground_truth.txt`) drops to 1–4 bpm for subjects 11, 18 and 20, so it is reported only as a
secondary number (`hr_mae_vs_oximeter_bpm`).

## Input normalization

`temporal` (default): subtract each pixel's mean over the window, then divide by one global
std. This removes the skin-colour DC term and static illumination so the ~1 % pulse
component is what the network sees, while keeping the relative AC amplitude between skin and
background pixels. `raw` (pixels / 255) is what the legacy checkpoint was trained on.

## Results

Held-out **test** split (7 subjects, 90 non-overlapping 5-s windows). HR reference = FFT
peak of the ground-truth BVP with harmonic-aware peak picking (`config.HR_SUBHARMONIC_*`).

| run | data pipeline | loss | MSE | Pearson r | HR MAE | HR RMSE | within 5 bpm | SNR | amp. ratio |
|---|---|---|---|---|---|---|---|---|---|
| legacy v2 (`best_rppg_model_v2.pth`) | 362 non-overlapping windows, raw pixels | MSE | 0.671 | 0.634 | 2.45 bpm | 6.83 | 92.2 % | 0.73 dB | 0.47 |
| **v2_newpipe** (`results/v2_newpipe/`) | stride 30 (1,754 windows), temporal norm, augmentation | MSE | **0.431** | **0.759** | **1.22 bpm** | **2.48** | **96.7 %** | **2.01 dB** | **0.73** |

Same model, same loss, same optimizer and selection rule — only the data pipeline changed.
Validation (used for early stopping, so optimistic): legacy r 0.710 / HR MAE 0.79 bpm;
v2_newpipe r 0.817 / HR MAE 0.90 bpm. v2_newpipe early-stopped at epoch 26 (best epoch 18),
~8.5–14 min/epoch on a 12-core laptop CPU.

Remaining test errors: subject12's first 10 s (irregular reference waveform) and one window
of subject20 with an HR ramp inside the window. Amplitude ratio 0.73 is the residual MSE
regression-to-the-mean; a correlation-based loss is the next step.

## Collaborator workflow (training on another machine)

The raw dataset (70 GB) and the preprocessing cache are **not** in git. Trained run
directories (`results/<run>/best.pth`, `last.pth`, `history.json`) **are** tracked, so a
model trained elsewhere can be pushed back and evaluated here.

```bash
git clone https://github.com/Joshika-joe/RPPG.git
cd RPPG
python -m venv .venv && .venv\Scripts\activate       # or: source .venv/bin/activate
pip install -r requirements.txt
# NVIDIA GPU? install the CUDA build instead of the CPU one, e.g.
#   pip install torch --index-url https://download.pytorch.org/whl/cu124
# get_device() picks CUDA automatically.

# 1. Put UBFC-rPPG at DATASET_2/subjectN/{vid.avi, ground_truth.txt}
# 2. Preprocess (~3.5 min on CPU)
python -m src.preprocessing

# 3. Train (a new run name; --resume continues an interrupted run from its last.pth)
python -m src.train --model v2 --run my_run

# 4. Evaluate on the held-out test split
python -m src.evaluate --run results/my_run --split test

# 5. Push the trained run + metrics back
git add results/my_run results/my_run_test
git commit -m "Add my_run"
git push
```

Training is CPU-heavy: ~12–16 min/epoch on a 12-core laptop with 400 windows/epoch; a GPU
brings this to well under a minute. Early stopping (patience 8 on val MSE) usually ends the
run around epoch 20–30.

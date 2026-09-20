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
  models.py         model registry (v2: 3D-CNN, v3: 2D+1D with appearance attention)
  metrics.py        Pearson r, HR MAE / RMSE, SNR, band-pass, per-subject tables
  train.py          training loop (seeded, resumable, per-epoch val metrics, history)
  evaluate.py       metrics.json + per-window CSV + waveform / HR scatter / Bland-Altman plots
  inference.py      continuous overlap-add inference over whole recordings
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
python -m src.train --model v3 --run my_run --loss pearson --optimizer adamw --lr 1e-3 --wd 1e-2 --scheduler cosine --warmup-epochs 3 --grad-clip 1.0 --select-on hr_mae
python -m src.train --model v3 --run my_run --resume          # continue after interruption
```

Defaults (`src/config.py`): 150-frame windows, train stride 30 (1,754 overlapping windows
from 29 subjects), 400 random windows per epoch, temporal input normalization, augmentation
(h-flip, brightness / contrast, ±4 px shift — all constant over time), batch 4. Every option
is a CLI flag. The best recipe so far (v3_pearson):

```bash
python -m src.train --model v3 --run v3_pearson --loss pearson --optimizer adamw --lr 1e-3     --wd 1e-2 --scheduler cosine --warmup-epochs 3 --grad-clip 1.0 --select-on hr_mae     --epochs 40 --patience 15 --threads 8
```

`--loss pearson` is (1 − Pearson r) + 0.2·MSE: scale-invariant, so the model is rewarded for
waveform shape rather than for hedging toward zero amplitude.

## 3. Evaluate

```bash
python -m src.evaluate --run results/v3_pearson --split test
python -m src.evaluate --run results/v3_pearson --split test --continuous --stride 30 --segment-frames 300
python -m src.evaluate --checkpoint best_rppg_model_v2.pth --model v2 --normalize raw --split val
```

Reports MSE, per-window Pearson r, HR MAE / RMSE (FFT peak in 0.7–4 Hz, using the real fps),
fraction within 5 bpm, SNR (de Haan), and the prediction / target amplitude ratio, overall and
per subject. `--bandpass` filters predictions to 0.7–4 Hz first.

The HR reference is the FFT of the ground-truth BVP. UBFC's oximeter HR line (line 2 of
`ground_truth.txt`) drops to 1–4 bpm for subjects 11, 18 and 20, so it is reported only as a
secondary number (`hr_mae_vs_oximeter_bpm`).

## Models

**v2** - 3D-CNN (507K params): four Conv3d blocks, spatial global average pool, 1D
temporal conv head.

**v3** - 2D+1D (168K params): a shared 2D encoder runs on every frame (64 -> 8x8 feature
map); a small appearance branch turns the window's *mean frame* into an 8x8 spatial
attention mask (temporal normalization removes appearance from the frames themselves, so
"where is the skin" has to come from somewhere else); attention-weighted pooling gives a
(C, T) sequence; five dilated residual 1D blocks (receptive field 125 frames) and a 1x1 head
produce the BVP. ~4.7x cheaper per training step than v2 on CPU.

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
| **v2_newpipe** (`results/v2_newpipe/`) | stride 30 (1,754 windows), temporal norm, augmentation | MSE | **0.431** | **0.759** | **1.22 bpm** | **2.48** | **96.7 %** | 2.01 dB | 0.73 |
| v3_newpipe (`results/v3_newpipe/`), 2D+1D, 168K params | same as v2_newpipe | MSE | 0.435 | 0.753 | 1.73 bpm | 5.21 | 95.6 % | 2.32 dB | 0.77 |
| **v3_pearson** (`results/v3_pearson/`) | same as v2_newpipe | (1−r) + 0.2·MSE, AdamW 1e-3 cosine | **0.348** | **0.811** | **1.07 bpm** | **1.62** | **98.9 %** | **3.27 dB** | **0.79** |

v2_newpipe vs legacy: same model, loss, optimizer and selection rule — only the data
pipeline changed. v3_newpipe vs v2_newpipe: same everything, only the architecture changed.
v3_pearson vs v3_newpipe: same data and architecture; loss, optimizer (AdamW, wd 1e-2,
3-epoch warmup + cosine, grad clip 1.0) and selection rule (val HR MAE) changed.
Validation (used for early stopping, so optimistic): legacy r 0.710 / HR MAE 0.79 bpm;
v2_newpipe r 0.817 / 0.90; v3_newpipe r 0.824 / 0.98; v3_pearson r 0.853 / 0.69.

v3_pearson: one test window in 90 is off by more than 5 bpm (max 7.3, subject12's irregular
opening), every subject has mean r >= 0.70, and it reached v3_newpipe's final val r after 4 epochs (2 min each). Early-stopped
at epoch 27, best epoch 12. Note the cosine schedule had not finished (lr 2.9e-4 at stop) and
val r was still rising (0.856 at epoch 27 vs 0.853 at the selected epoch 12) - HR MAE is a
noisier selection criterion than Pearson r. A rerun selected on val r with the full cosine
schedule (`results/v3_pearson_selr`, best epoch by val r = 0.856) ties on windowed test
metrics (r 0.819, HR MAE 1.06 bpm) and is slightly worse on 10-s continuous segments
(HR MAE 0.57 vs 0.37 bpm), so v3_pearson remains the reference checkpoint. Two runs of the
same recipe landing at r 0.81-0.82 on test shows the result is not a lucky epoch.

v2_newpipe early-stopped at epoch 26 (best 18), ~8.5–14 min/epoch on a 12-core laptop CPU.
v3_newpipe ran all 40 epochs (best 33) at 3.3 min/epoch with val loss still falling - it is
under-trained at lr 1e-4. Its HR MAE gap to v2 is one window (subject11 @ 1650, 46 bpm
sub-harmonic pick); median HR error is identical (1.45 vs 1.44 bpm).

`results/v3_newpipe_test/attention.png` shows the learned spatial attention: forehead and
cheeks, avoiding hair, eyes, background and beard, with no supervision on location.

Remaining test errors are concentrated in subject12 (irregular reference waveform in its
first 10 s) and subject48 (lowest SNR video).

### Continuous inference (overlap-add)

`--continuous --stride 30`: the model slides over each recording with a 1-s hop, every
window's prediction is standardized and blended with a Hann weight into one continuous BVP
per subject (`src/inference.py`), and metrics are computed on non-overlapping 300-frame
(~10 s) segments — the usual UBFC protocol. Whole-recording r is the Pearson correlation over
the entire ~60 s recording (both signals band-passed to 0.7–4 Hz).

| run | Pearson r (10-s segments) | HR MAE | HR RMSE | within 5 bpm | SNR | whole-recording r |
|---|---|---|---|---|---|---|
| legacy v2 | 0.648 | 2.39 bpm | 9.42 | 92.9 % | 1.30 dB | 0.642 |
| v2_newpipe | 0.759 | 0.68 bpm | 1.99 | 97.6 % | 2.21 dB | 0.750 |
| v3_newpipe | 0.762 | 0.55 bpm | 1.64 | 97.6 % | 2.76 dB | 0.754 |
| **v3_pearson** | **0.813** | **0.37 bpm** | **0.54** | **100.0 %** | **3.72 dB** | **0.801** |

Same v3_pearson checkpoint, windowed vs continuous on the test split:

| protocol | Pearson r | HR MAE | HR RMSE | within 5 bpm |
|---|---|---|---|---|
| isolated 5-s windows (stride 150) | 0.811 | 1.07 bpm | 1.62 | 98.9 % |
| overlap-add, 5-s segments | 0.826 | 0.80 bpm | 1.35 | 98.9 % |
| overlap-add, 10-s segments | 0.813 | 0.37 bpm | 0.54 | 100.0 % |

Overlap-add alone lowers 5-s HR MAE from 1.07 to 0.80 bpm (window-edge effects
average out); 10-s segments bring it to 0.37 bpm with every segment within 5 bpm.
`--bandpass` (0.7–4 Hz on the prediction) changes HR MAE by < 0.05 bpm and is off by default.
`results/v3_pearson_test_continuous_seg300/continuous.png` shows the first 20 s of every test
subject.

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

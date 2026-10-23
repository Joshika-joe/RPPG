# Cross-dataset evaluation (Phase 8)

Everything reported so far comes from UBFC-rPPG. A model that scores well on a split of its
own dataset has not been shown to generalise, so this phase evaluates the existing
checkpoint on a genuinely different recording distribution **without retraining**.

**Status: the adapter and evaluation pipeline are implemented and tested on synthetic data.
Real cross-dataset performance is UNMEASURED, because neither candidate dataset can be
downloaded from this environment — both require a human to accept terms or request access.**

## Candidate comparison

| | **PURE** | **UBFC-Phys** |
|---|---|---|
| Subjects / recordings | 10 subjects, 60 sequences (6 setups x 1 min) | 56 participants, 3 tasks each (168 recordings) |
| Video | eco274CVGE, 640x480, 30 Hz, 4.8 mm lens | Edmund Optics EO-23121C, 1024x1024, 35 fps, `.avi` |
| Reference | pulox CMS50E **finger** pulse oximeter, waveform + SpO2 at 60 Hz | Empatica E4 **wrist** band, BVP at 64 Hz (+ EDA) |
| Reference timestamps | published with the oximeter stream | not published per frame; derived from the container rate |
| Conditions vs UBFC-rPPG | six deliberate head-motion setups (steady, talking, slow/fast translation, small/medium rotation) | social-stress protocol: rest (T1), speech (T2), arithmetic (T3) |
| File layout | not stated on the official page | `s<N>/vid_s<N>_T<k>.avi`, `s<N>/bvp_s<N>_T<k>.csv` |
| Access | **email request to `nikr-datasets-request@tu-ilmenau.de`** | **IEEE DataPort open access**, free IEEE account |
| Access restriction | **academic institutions only, from an institutional email address. Personal providers such as Gmail are explicitly not processed; companies are excluded** | account registration; cite the paper |
| Can the existing model be evaluated without retraining? | yes | yes |
| Preprocessing adaptation needed | image-sequence reader + JSON reference parser (format not confirmed from official docs) | `.avi` + `.csv` reader — the existing OpenCV path already handles both |

### Chosen dataset: UBFC-Phys

1. **Access is realistically obtainable.** PURE requires an institutional academic email and
   states that requests from personal providers such as Gmail are not processed. UBFC-Phys is
   open access on IEEE DataPort and needs only a free account.
2. **No new readers required.** `.avi` video and a one-value-per-line `.csv` reference both
   fit the existing OpenCV pipeline, so the adaptation is genuinely input-format only.
3. **It is a harder and more informative distribution shift.** 56 participants rather than 10,
   a different camera at a different resolution and frame rate, and — unlike UBFC-rPPG, where
   everyone sits still — two of the three tasks involve speaking and mental arithmetic, so
   facial motion is present by design.

PURE remains worth doing if you have a university email address; the same evaluator works
once an adapter for its format is written.

### What UBFC-Phys cannot tell us

* **The reference is a wrist sensor.** The pulse reaches the wrist later than the face
  (pulse transit time), and the E4 is not hardware-synchronised with the camera, so an unknown
  constant offset may exist. **Heart rate is unaffected by a constant lag; waveform
  correlation is.** `src/eval_cross.py` therefore reports correlation at zero lag *and* the
  best correlation within +/- 1 s, labelled a diagnostic and never used to select anything.
* **No per-frame timestamps are published.** Frame times are derived from the container rate.
  UBFC-rPPG ships real, irregular timestamps; this dataset does not, so the two are evaluated
  under different timing conventions and that difference is recorded in each store's
  `meta.json` (`"timestamps"`).
* **No reference heart-rate channel.** Only the BVP waveform is provided, so the reference HR
  is derived from it by the same FFT estimator used on the prediction — the same convention
  already used for UBFC-rPPG, where the published oximeter HR line is unreliable.

## What you need to do to obtain the data

1. Create a free account at `ieee-dataport.org`.
2. Open the UBFC-Phys dataset page (DOI **10.21227/5da0-7344**) and accept whatever terms the
   page presents, then download the archive.
3. Extract it so the layout is:

```
<UBFC-Phys>/
  s1/
    vid_s1_T1.avi   vid_s1_T2.avi   vid_s1_T3.avi
    bvp_s1_T1.csv   bvp_s1_T2.csv   bvp_s1_T3.csv
  s2/
    ...
```

Only `vid_*.avi` and `bvp_*.csv` are used; `eda_*` and `info_*` files are ignored. A subset is
fine — the pipeline discovers whatever participants and tasks are present.

Cite: *UBFC-Phys: A Multimodal Database For Psychophysiological Studies Of Social Stress*,
IEEE Transactions on Affective Computing, 2021.

## Reproduction commands

```bash
# 1. Adapt the dataset into the project's store format (separate from the UBFC-rPPG cache).
python -m src.ubfc_phys --root /path/to/UBFC-Phys

#    Start with the rest task and a few participants to check it end to end:
python -m src.ubfc_phys --root /path/to/UBFC-Phys --tasks T1 --participants s1 s2 s3

# 2. Zero-shot evaluation with the existing checkpoint, estimator, window and thresholds.
python -m src.eval_cross --run results/v3_pearson \
    --subject-root checkpoints/subjects_ubfc_phys --label ubfc_phys

#    Per-task breakdown (rest vs speech vs arithmetic) comes out of the same command.

# 3. Diagnose the worst cases, exactly as for UBFC-rPPG.
python -m src.failure_analysis --run results/v3_pearson --split val   # UBFC-rPPG reference

# 4. Tests (synthetic; no dataset needed).
python -m pytest tests/test_ubfc_phys.py -q
```

Step 1 writes to `checkpoints/subjects_ubfc_phys/`, which is separate from
`checkpoints/subjects/`; step 2 writes to `results/crossdataset_ubfc_phys/`. No existing
UBFC-rPPG cache or result is modified.

Expect step 1 to take roughly as long as decoding the videos: UBFC-Phys clips are
1024x1024, about four times the pixels of UBFC-rPPG, so budget a few minutes per ten
recordings on a laptop CPU.

## How the UBFC-rPPG numbers should be labelled in any comparison

The existing UBFC-rPPG test figures are **development results, not an untouched benchmark**:

* The test split has informed several decisions across phases 1-7 (it was reported after each
  phase, and during the estimator comparison).
* UBFC-rPPG timestamps are irregular; honouring them moves the *reference* heart rate by a
  mean of 1.32 bpm (max 5.18) on the test split. The reported agreement of ~0.4 bpm is a
  measure of model-vs-reference agreement on a shared time axis, **not** physiological
  accuracy, which that dataset cannot resolve below about 1.3 bpm.

Any cross-dataset difference must therefore not be attributed to the network alone:
acquisition hardware, resolution, frame rate, reference sensor site and synchronisation all
differ, and each can move heart-rate error on its own.

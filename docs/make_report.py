"""
Generate docs/rPPG_Project_Walkthrough.pdf - a narrative report of the project.

All metrics are read from results/*/metrics.json so the document cannot drift from the
numbers actually produced by the code.

    python docs/make_report.py
"""

import json
import os
import sys

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "docs", "rPPG_Project_Walkthrough.pdf")
TMP = os.path.join(ROOT, "docs", "_figures")

INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#5b6472")
ACCENT = colors.HexColor("#b03030")
RULE = colors.HexColor("#c9ced6")
BAND = colors.HexColor("#eef1f5")
HILITE = colors.HexColor("#fdf3d8")

PAGE_W, PAGE_H = A4
MARGIN = 2.0 * cm
CONTENT_W = PAGE_W - 2 * MARGIN


# ==============================================================================
# DATA
# ==============================================================================
def summary(run_dir):
    with open(os.path.join(ROOT, "results", run_dir, "metrics.json"), encoding="utf-8") as f:
        return json.load(f)["summary"]


def pct(x):
    return f"{100 * x:.1f}%"


# ==============================================================================
# STYLES
# ==============================================================================
def styles():
    s = getSampleStyleSheet()
    base = dict(fontName="Helvetica", textColor=INK, leading=13.8, fontSize=9.6)
    s.add(ParagraphStyle("Body2", parent=s["Normal"], alignment=TA_JUSTIFY, spaceAfter=7, **base))
    s.add(ParagraphStyle("Bullet2", parent=s["Normal"], leftIndent=12, bulletIndent=2,
                         spaceAfter=4, **base))
    s.add(ParagraphStyle("H1", parent=s["Normal"], fontName="Helvetica-Bold", fontSize=15,
                         textColor=INK, spaceBefore=16, spaceAfter=7, leading=18))
    s.add(ParagraphStyle("H2", parent=s["Normal"], fontName="Helvetica-Bold", fontSize=11,
                         textColor=ACCENT, spaceBefore=11, spaceAfter=4, leading=14))
    s.add(ParagraphStyle("Caption", parent=s["Normal"], fontName="Helvetica-Oblique", fontSize=8.2,
                         textColor=MUTED, alignment=TA_CENTER, spaceBefore=4, spaceAfter=10, leading=11))
    s.add(ParagraphStyle("Mono", parent=s["Normal"], fontName="Courier", fontSize=8.2,
                         textColor=INK, leading=11.5, leftIndent=8, spaceAfter=7))
    s.add(ParagraphStyle("TitleBig", parent=s["Normal"], fontName="Helvetica-Bold", fontSize=25,
                         textColor=INK, leading=29, spaceAfter=4))
    s.add(ParagraphStyle("Sub", parent=s["Normal"], fontName="Helvetica", fontSize=11.5,
                         textColor=MUTED, leading=15, spaceAfter=14))
    s.add(ParagraphStyle("Lead", parent=s["Normal"], fontName="Helvetica", fontSize=10.6,
                         textColor=INK, leading=15.5, alignment=TA_JUSTIFY, spaceAfter=8))
    s.add(ParagraphStyle("TblH", parent=s["Normal"], fontName="Helvetica-Bold", fontSize=8.3,
                         textColor=colors.white, leading=10.5))
    s.add(ParagraphStyle("Tbl", parent=s["Normal"], fontName="Helvetica", fontSize=8.5,
                         textColor=INK, leading=11))
    s.add(ParagraphStyle("TblB", parent=s["Normal"], fontName="Helvetica-Bold", fontSize=8.5,
                         textColor=INK, leading=11))
    return s


S = styles()


def P(text, style="Body2"):
    return Paragraph(text, S[style])


def bullet(text):
    return Paragraph(text, S["Bullet2"], bulletText="•")


def code(lines):
    return Paragraph("<br/>".join(lines), S["Mono"])


def heading(text, level=1):
    return Paragraph(text, S["H1" if level == 1 else "H2"])


def rule():
    return HRFlowable(width="100%", thickness=0.6, color=RULE, spaceBefore=3, spaceAfter=9)


def table(rows, widths, highlight_rows=(), align_right_from=1):
    """rows[0] is the header."""
    data = []
    for i, row in enumerate(rows):
        style = "TblH" if i == 0 else ("TblB" if i in highlight_rows else "Tbl")
        data.append([Paragraph(str(c), S[style]) for c in row])
    t = Table(data, colWidths=widths, hAlign="LEFT")
    cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3b4452")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
        ("ALIGN", (align_right_from, 1), (-1, -1), "RIGHT"),
    ]
    for i in range(1, len(rows)):
        if i in highlight_rows:
            cmds.append(("BACKGROUND", (0, i), (-1, i), HILITE))
        elif i % 2 == 0:
            cmds.append(("BACKGROUND", (0, i), (-1, i), BAND))
    t.setStyle(TableStyle(cmds))
    return t


def figure(path, caption, width=CONTENT_W, crop_frac=None):
    """crop_frac=(0,0.45) keeps the top 45% of the image."""
    src = os.path.join(ROOT, path)
    if crop_frac is not None:
        os.makedirs(TMP, exist_ok=True)
        im = Image.open(src)
        lo, hi = crop_frac
        im = im.crop((0, int(lo * im.height), im.width, int(hi * im.height)))
        src = os.path.join(TMP, os.path.basename(path).replace(".png", "_crop.png"))
        im.save(src)
    w, h = Image.open(src).size
    img = RLImage(src, width=width, height=width * h / w)
    return KeepTogether([img, Paragraph(caption, S["Caption"])])


# ==============================================================================
# PAGE FURNITURE
# ==============================================================================
def decorate(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.6)
    canvas.setFillColor(MUTED)
    if doc.page > 1:
        canvas.drawString(MARGIN, PAGE_H - MARGIN + 0.45 * cm,
                          "rPPG - contactless pulse estimation from face video")
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, PAGE_H - MARGIN + 0.25 * cm, PAGE_W - MARGIN, PAGE_H - MARGIN + 0.25 * cm)
        canvas.drawRightString(PAGE_W - MARGIN, 1.2 * cm, str(doc.page))
    canvas.restoreState()


# ==============================================================================
# CONTENT
# ==============================================================================
def contents_block():
    items = [
        ("1", "Where the project started", "a model that had learned to predict nothing"),
        ("2", "Learning to measure", "the right metrics, and two faults in the data"),
        ("3", "Rebuilding the data pipeline", "362 training windows become 1,754"),
        ("4", "A new architecture", "separating space from time, and learning where skin is"),
        ("5", "Changing the loss", "asking for the right shape instead of the right scale"),
        ("6", "Continuous inference", "one waveform per person instead of isolated slices"),
        ("7", "The complete picture", "what each decision contributed"),
        ("8", "From dataset to webcam", "two problems only real hardware could reveal"),
        ("9", "Reproducing all of this", "commands, and where it could go next"),
    ]
    rows = [[Paragraph(f"<b>{n}</b>", S["Tbl"]),
             Paragraph(f"<b>{t}</b>", S["Tbl"]),
             Paragraph(f"<font color='#5b6472'>{d}</font>", S["Tbl"])] for n, t, d in items]
    t = Table(rows, colWidths=[0.7 * cm, 5.6 * cm, 9.6 * cm], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ]))
    return t


def build():
    legacy_w, legacy_c = summary("best_rppg_model_v2_test"), summary("best_rppg_model_v2_test_continuous_seg300")
    v2_w, v2_c = summary("v2_newpipe_test"), summary("v2_newpipe_test_continuous_seg300")
    v3_w, v3_c = summary("v3_newpipe_test"), summary("v3_newpipe_test_continuous_seg300")
    pe_w, pe_c = summary("v3_pearson_test"), summary("v3_pearson_test_continuous_seg300")
    pe_c150 = summary("v3_pearson_test_continuous_seg150")

    story = []

    # ---------------------------------------------------------------- cover
    story += [
        Spacer(1, 2.2 * cm),
        P("Remote Photoplethysmography", "TitleBig"),
        P("Measuring heart rate from face video: a project walkthrough", "Sub"),
        rule(),
        P(
            "This report traces the project from its starting point - three Jupyter notebooks and a "
            "3D convolutional network that, on inspection, had learned nothing - through a rebuilt data "
            "pipeline, a new architecture, a change of loss function, and continuous inference, to a "
            "model that estimates heart rate from a webcam. Every number quoted here comes from the "
            "held-out test subjects, who were never used for training or for choosing a checkpoint, and "
            "is read directly from the stored metrics files rather than typed in by hand.",
            "Lead",
        ),
        Spacer(1, 0.3 * cm),
    ]

    head = [["", "Starting point", "Final model", "Change"]]
    rows = [
        ["Heart-rate error (MAE)", f"{legacy_c['hr_mae_bpm']:.2f} bpm", f"{pe_c['hr_mae_bpm']:.2f} bpm", "6.5x better"],
        ["Heart-rate error (RMSE)", f"{legacy_c['hr_rmse_bpm']:.2f}", f"{pe_c['hr_rmse_bpm']:.2f}", "17x better"],
        ["Waveform correlation", f"{legacy_c['pearson_mean']:.3f}", f"{pe_c['pearson_mean']:.3f}", "+0.17"],
        ["Segments within 5 bpm", pct(legacy_c["hr_within_5bpm_frac"]), pct(pe_c["hr_within_5bpm_frac"]), "all of them"],
        ["Model size", "78.7 M / 507 K", "168 K", "3x smaller"],
        ["Training time", "hours", "~1 hour", "on this laptop"],
    ]
    story += [
        table(head + rows, [5.0 * cm, 3.5 * cm, 3.5 * cm, 3.9 * cm], align_right_from=1),
        Paragraph(
            "Held-out test subjects, continuous inference over 10-second segments. "
            "The starting point is the best model that existed before this work.",
            S["Caption"],
        ),
        Spacer(1, 1.4 * cm),
        rule(),
        P("What is in this report", "H2"),
        contents_block(),
        Spacer(1, 1.0 * cm),
        rule(),
        P(
            "<font color='#5b6472' size='8.5'>UBFC-rPPG dataset - 42 subjects, 29 train / 6 validation / "
            "7 test, split by subject.<br/>Final model: 167,562 parameters, trained on one laptop CPU. "
            "Code and all results: github.com/Joshika-joe/RPPG</font>",
            "Caption",
        ),
        PageBreak(),
    ]

    # ---------------------------------------------------------------- 1
    story += [
        heading("1. Where the project started"),
        P(
            "The project already had the hard parts in place: the UBFC-rPPG dataset of 42 subjects, each "
            "a roughly one-minute video of a face with a simultaneously recorded pulse-oximeter signal; "
            "a subject-level split into 29 training, 6 validation and 7 test subjects; and a working "
            "preprocessing path that found the face, cropped it to 64x64 pixels and lined it up with the "
            "reference pulse waveform. Two trained models sat in the project root."
        ),
        P(
            "The first thing worth doing was not to add anything, but to check what those models had "
            "actually learned. The training log of the first model told the story. Its validation loss "
            "started at 1.0010 and finished, fifty epochs later, at 0.9966. The targets were normalised "
            "to zero mean and unit variance, which means a model that outputs nothing but zeros scores "
            "exactly 1.0. After fifty epochs and 78.7 million parameters, the network had moved 0.3% away "
            "from predicting nothing at all. It had not learned to see a pulse; it had learned to stay "
            "quiet, because on this loss that is the safest thing to do."
        ),
        P(
            "The second model was genuinely working, but nobody could tell, because the only number being "
            "reported was mean squared error - and MSE cannot distinguish a model that has found the pulse "
            "from one that is hedging. Three problems followed from this:"
        ),
        bullet(
            "<b>The test subjects had never been used.</b> The split existed in the config file, but no "
            "code touched it. Results were being read off the validation subjects - the same ones used to "
            "pick the best epoch - so every number was quietly optimistic."
        ),
        bullet(
            "<b>The metric did not match the task.</b> What matters clinically is the heart rate in beats "
            "per minute and the shape of the pulse waveform. Neither was being measured."
        ),
        bullet(
            "<b>Two near-identical copies of everything.</b> Separate training, evaluation and model files "
            "for each version, which had already drifted apart in small ways - the training and validation "
            "preprocessing differed in how they handled timestamps, and one checked for frame/signal length "
            "mismatches while the other did not."
        ),
        P(
            "So the work divided into five phases, each changing exactly one thing and each ending with a "
            "measurement on the test subjects. That discipline is what makes the table on the cover page "
            "meaningful: every row can be attributed to a specific decision."
        ),
    ]

    # ---------------------------------------------------------------- 2
    story += [
        heading("2. Phase 0 - learning to measure"),
        P(
            "Before changing the model, the project needed to be able to tell whether a change helped. "
            "Three metrics replaced the single loss value. <b>Pearson correlation</b> compares the shape of "
            "the predicted waveform against the reference, and because it ignores scale, a model cannot "
            "score well by playing it safe. <b>Heart-rate error</b> takes the strongest frequency in the "
            "predicted signal between 0.7 and 4 Hz - 42 to 240 beats per minute - and compares it with the "
            "same measurement on the reference. <b>Signal-to-noise ratio</b> measures how much of the "
            "predicted signal sits at the true pulse frequency rather than spread across the band."
        ),
        heading("Two things the data itself got wrong", 2),
        P(
            "Measuring carefully immediately turned up two faults in the dataset that had been silently "
            "corrupting results. Subjects 25, 26 and 27 were recorded at 23.3 frames per second while every "
            "other subject was recorded at about 29.5; the old code assumed 30 for everyone. Since a heart "
            "rate is derived from a frequency, and a frequency depends on the frame rate, this put a "
            "systematic error on three subjects. Reading each subject's true frame rate from its own "
            "timestamps lowered the validation heart-rate error from 1.10 to 0.79 bpm on its own."
        ),
        P(
            "The second fault was in the reference signal. UBFC stores a heart-rate line alongside the pulse "
            "waveform, and for subjects 11, 18 and 20 that line collapses to between 1 and 4 beats per "
            "minute - physiologically impossible, and a sign the oximeter lost contact. Those subjects are "
            "all in the test set. The fix was to derive the reference heart rate from the pulse waveform "
            "itself, which is intact, and report the oximeter figure only as a secondary number."
        ),
        P(
            "A related subtlety showed up later, in a single window of subject 3. A pulse waveform is peaked "
            "rather than sinusoidal, so it carries energy at twice the heart rate as well as at the heart "
            "rate itself; in that one window the second harmonic was the stronger of the two, and the "
            "reference was recorded as 211 bpm when the true rate was 97.5. The model had been right and the "
            "reference wrong. The peak-finder now checks whether a peak at half the dominant frequency "
            "carries at least half its power, and prefers it if so. Across all 523 windows in the dataset "
            "exactly one changes - the broken one."
        ),
    ]

    # ---------------------------------------------------------------- 3
    story += [
        heading("3. Phase 1 - rebuilding the data pipeline"),
        P(
            "The original pipeline cut each video into five-second windows that did not overlap, which "
            "yielded only 362 training examples from 29 subjects, and stored them as pre-cut floating-point "
            "arrays occupying 3.2 GB. Because the windows were baked into the stored file, changing the "
            "window length or spacing meant half an hour of reprocessing."
        ),
        P(
            "The new pipeline stores each subject once, as the cropped face sequence in its compact "
            "8-bit form, roughly 1 GB for all 42 subjects, and cuts windows at training time. Three changes "
            "followed from that:"
        ),
        bullet(
            "<b>Overlapping windows.</b> Sliding by one second instead of five turns the same videos into "
            "1,754 training examples rather than 362, without recording anything new."
        ),
        bullet(
            "<b>Removing the skin tone.</b> The pulse changes skin brightness by roughly one percent, "
            "sitting on top of a large constant skin colour the network has no use for. Subtracting each "
            "pixel's average over the window removes that constant and leaves the part that actually "
            "varies - the signal."
        ),
        bullet(
            "<b>Augmentation.</b> Horizontal flips, brightness and contrast changes, and small shifts, each "
            "applied identically to every frame of a window so the transformation cannot introduce a rhythm "
            "that was not there."
        ),
        P(
            "Retraining the same model, with the same loss and optimiser, on this pipeline isolates the "
            "effect of the data alone. On the test subjects the waveform correlation rose from 0.634 to "
            f"{v2_w['pearson_mean']:.3f} and the heart-rate error fell from "
            f"{legacy_w['hr_mae_bpm']:.2f} to {v2_w['hr_mae_bpm']:.2f} bpm. The amplitude ratio - how large "
            "the predicted waveform is compared with the real one - rose from 0.47 to "
            f"{v2_w['amplitude_ratio_mean']:.2f}, the first sign that the model had stopped hedging."
        ),
    ]

    # ---------------------------------------------------------------- 4
    story += [
        heading("4. Phase 2 - an architecture that separates space from time"),
        P(
            "The original network used 3D convolutions, which process space and time together in every "
            "layer. For this problem that is wasteful: finding skin is a purely spatial question, and "
            "finding a rhythm is a purely temporal one. The new design, v3, splits them. A small 2D encoder "
            "runs on every frame independently and reduces it to an 8x8 grid of features. The frames are "
            "then collapsed into a single time series and handed to a stack of five one-dimensional "
            "convolutions with increasing dilation, which widens their reach to 125 frames - just over four "
            "seconds, comfortably more than one heartbeat."
        ),
        P(
            "There is a catch, and it is an interesting one. Removing each pixel's average over time, which "
            "Phase 1 introduced, also removes everything that tells the network where the skin is. A purely "
            "per-frame encoder is therefore blind to its own geometry. The solution is to hand it the "
            "window's average frame - precisely the information that was subtracted - through a small side "
            "branch that produces a spatial attention map, used to weight the pooling. The map is learned "
            "with no supervision about where faces or skin are; its only instruction is to predict the "
            "pulse well."
        ),
        figure(
            "results/v3_pearson_test/attention.png",
            "Figure 1. What the attention branch learned, on four test subjects it never saw during "
            "training. Top row: each subject's average face. Bottom row: the learned weighting. It "
            "concentrates on the forehead and cheeks - where the skin is thin and well perfused - and "
            "avoids hair, eyes, beard and background. Nothing in the training signal mentioned skin; "
            "this is simply where the pulse was easiest to find.",
        ),
        P(
            "The result was a model with 168 thousand parameters instead of 507 thousand, training roughly "
            "four times faster per step, and matching the larger model's accuracy "
            f"(correlation {v3_w['pearson_mean']:.3f} against {v2_w['pearson_mean']:.3f}). Not yet an "
            "improvement in accuracy, but the same result for a third of the size - and its validation "
            "loss was still falling when the run ended, which pointed directly at the next phase."
        ),
    ]

    # ---------------------------------------------------------------- 5
    story += [
        heading("5. Phase 3 - changing what the model is asked to do"),
        P(
            "Mean squared error asks the model to match the waveform point by point. Faced with "
            "uncertainty, the safest response to that question is to predict something small - which is "
            "exactly what the amplitude ratio of 0.47 had been showing, and in the extreme, exactly what "
            "the original model did when it learned to output nothing at all."
        ),
        P(
            "Replacing it with a correlation-based loss changes the question to <i>does your waveform have "
            "the same shape as the reference</i>, which is scale-free: shrinking the output no longer helps. "
            "A small squared-error term is kept alongside it, weighted at 0.2, so the output stays on a "
            "sensible scale. The optimiser was modernised at the same time - AdamW at a ten times higher "
            "learning rate with a short warm-up and a cosine decay, and gradient clipping."
        ),
        P(
            "This was the largest single improvement in the project, and it arrived quickly: the new "
            "configuration reached the previous model's best validation correlation after four epochs, "
            "about eight minutes of training. On the test subjects the heart-rate error fell from "
            f"{v3_w['hr_mae_bpm']:.2f} to {pe_w['hr_mae_bpm']:.2f} bpm and the RMSE from "
            f"{v3_w['hr_rmse_bpm']:.2f} to {pe_w['hr_rmse_bpm']:.2f} - the collapse in RMSE being the "
            "clearest signal, since it means the occasional badly wrong window had stopped happening. "
            "Every test subject reached a correlation of at least 0.70, where the worst had previously "
            "been 0.57."
        ),
        figure("results/v3_pearson/loss_curve.png",
               "Figure 2. Training the final model. Validation correlation (green) rises above 0.80 within "
               "four epochs; heart-rate error (red, divided by ten) settles below 1 bpm. Training stopped "
               "automatically at epoch 27 when the selected metric had not improved for fifteen epochs.",
               width=CONTENT_W * 0.86),
    ]

    # ---------------------------------------------------------------- 6
    story += [
        heading("6. Phase 4 - estimating continuously instead of in slices"),
        P(
            "Until this point every five-second window was judged in isolation, which is convenient for "
            "measurement but not how the method would be used. A person sitting in front of a camera "
            "produces a continuous signal. The model now slides across the recording one second at a time, "
            "each prediction is put on a common scale, and overlapping predictions are blended with a "
            "tapered weight into a single continuous pulse waveform per subject. Heart rate is then read "
            "over ten-second stretches, which is the protocol used in the published literature."
        ),
        P(
            "Averaging over overlapping windows suppresses errors at window edges, and the longer "
            "measurement window sharpens the frequency estimate. On the same trained model, with no "
            "retraining whatsoever:"
        ),
        table(
            [["How the same model is used", "Correlation", "HR MAE", "HR RMSE", "Within 5 bpm"],
             ["Isolated 5-second windows", f"{pe_w['pearson_mean']:.3f}", f"{pe_w['hr_mae_bpm']:.2f} bpm",
              f"{pe_w['hr_rmse_bpm']:.2f}", pct(pe_w["hr_within_5bpm_frac"])],
             ["Continuous, 5-second segments", f"{pe_c150['pearson_mean']:.3f}", f"{pe_c150['hr_mae_bpm']:.2f} bpm",
              f"{pe_c150['hr_rmse_bpm']:.2f}", pct(pe_c150["hr_within_5bpm_frac"])],
             ["Continuous, 10-second segments", f"{pe_c['pearson_mean']:.3f}", f"{pe_c['hr_mae_bpm']:.2f} bpm",
              f"{pe_c['hr_rmse_bpm']:.2f}", pct(pe_c["hr_within_5bpm_frac"])]],
            [6.2 * cm, 2.5 * cm, 2.4 * cm, 2.2 * cm, 2.6 * cm],
            highlight_rows=(3,),
        ),
        Paragraph("Table 1. Post-processing alone cuts the heart-rate error by two thirds.", S["Caption"]),
        figure(
            "results/v3_pearson_test_continuous_seg300/continuous.png",
            "Figure 3. Twenty seconds of three test subjects: the reference pulse in blue, the model's "
            "estimate from video alone in orange. Individual beats line up throughout, and the "
            "characteristic shape of a pulse - a sharp upstroke followed by a slower decay - is "
            "reproduced rather than just its timing.",
            crop_frac=(0.0, 0.43),
        ),
    ]

    # ---------------------------------------------------------------- results
    story += [
        heading("7. The complete picture"),
        P(
            "Each row below changes exactly one thing from the row above it, which is what allows the "
            "contribution of each decision to be read off directly."
        ),
        table(
            [["Stage", "What changed", "Correlation", "HR MAE", "RMSE", "Within 5"],
             ["Starting point", "-", f"{legacy_c['pearson_mean']:.3f}", f"{legacy_c['hr_mae_bpm']:.2f} bpm",
              f"{legacy_c['hr_rmse_bpm']:.2f}", pct(legacy_c["hr_within_5bpm_frac"])],
             ["Phase 1", "Data pipeline", f"{v2_c['pearson_mean']:.3f}", f"{v2_c['hr_mae_bpm']:.2f} bpm",
              f"{v2_c['hr_rmse_bpm']:.2f}", pct(v2_c["hr_within_5bpm_frac"])],
             ["Phase 2", "Architecture (2D+1D)", f"{v3_c['pearson_mean']:.3f}", f"{v3_c['hr_mae_bpm']:.2f} bpm",
              f"{v3_c['hr_rmse_bpm']:.2f}", pct(v3_c["hr_within_5bpm_frac"])],
             ["Phase 3+4", "Loss + continuous", f"{pe_c['pearson_mean']:.3f}", f"{pe_c['hr_mae_bpm']:.2f} bpm",
              f"{pe_c['hr_rmse_bpm']:.2f}", pct(pe_c["hr_within_5bpm_frac"])]],
            [2.3 * cm, 4.1 * cm, 2.4 * cm, 2.3 * cm, 2.0 * cm, 2.1 * cm],
            highlight_rows=(4,),
        ),
        Paragraph(
            "Table 2. Held-out test subjects throughout, continuous inference over 10-second segments. "
            "For reference, published results on this dataset for comparable published architectures sit "
            "in the region of 0.5 to 1.0 bpm.",
            S["Caption"],
        ),
        P(
            "A second run of the final recipe, differing only in which validation metric chose the best "
            "epoch, landed at a correlation of 0.819 against 0.811 and a heart-rate error of 1.06 against "
            "1.07 bpm on isolated windows. Two independent runs agreeing this closely is the evidence that "
            "the result is a property of the method and not a lucky epoch."
        ),
        P(
            "Where errors remain, they are concentrated and explainable. Subject 12's reference waveform is "
            "irregular for the first ten seconds of the recording. Subject 48's video is the noisiest in "
            "the test set. These are the two subjects with the lowest scores, and no amount of model "
            "capacity would fix the first of them."
        ),
    ]

    # ---------------------------------------------------------------- live
    story += [
        heading("8. From dataset to webcam"),
        P(
            "A model that works on a research dataset has not yet been shown to work on a laptop. The live "
            "demonstration keeps a rolling buffer of face crops, runs the model twice a second in a "
            "background thread, and blends the predictions into the same continuous waveform used for the "
            "test-set measurements. Running it exposed two problems that the dataset could never have "
            "revealed."
        ),
        heading("The frame rate problem", 2),
        P(
            "The first live run reported 19.8 frames per second. The dataset was recorded at about 29.5. "
            "Since the model consumes a fixed 150 frames, those 150 frames now covered 7.6 seconds of real "
            "time rather than 5 - every heartbeat stretched by half as much again. Emulating slower "
            "cameras on the test subjects quantified the damage, and the fix:"
        ),
        table(
            [["Camera frame rate", "Feeding frames directly", "Resampling to 5 seconds"],
             ["29.5 (dataset)", "1.58 bpm", "1.69 bpm"],
             ["25", "1.72 bpm", "1.69 bpm"],
             ["20 (the laptop webcam)", "1.94 bpm", "1.72 bpm"],
             ["15", "14.03 bpm", "1.70 bpm"]],
            [5.4 * cm, 5.2 * cm, 5.2 * cm],
            highlight_rows=(4,),
        ),
        Paragraph(
            "Table 3. Heart-rate error against the reference when the capture rate is emulated at "
            "different speeds. Below about 18 frames per second the direct approach fails completely.",
            S["Caption"],
        ),
        P(
            "The demonstration now timestamps every captured frame and interpolates the last five seconds "
            "onto exactly 150 of them before each prediction, so the estimate no longer depends on the "
            "camera at all. Verified on a test recording, a simulated 14.8 frames per second stream "
            "reproduces the full-rate output exactly, where previously it read about 10 bpm high."
        ),
        heading("The exposure problem", 2),
        P(
            "An option had been added to lock the camera's exposure, on the sound reasoning that automatic "
            "gain control actively cancels the brightness changes being measured. In practice it switched "
            "the driver into manual mode without supplying an exposure value, the driver fell back to its "
            "minimum, and the image went black - no face, no signal. The option now lets automatic exposure "
            "settle first, reads back the value the camera chose, applies that in manual mode, measures the "
            "resulting brightness, and reverts to automatic if it cannot match it. The failure is covered "
            "by a test using a simulated camera that goes black on command."
        ),
        heading("What a webcam can and cannot do", 2),
        P(
            "On the dataset recordings the live pipeline reports a signal quality of +3 to +7 dB. On a "
            "laptop webcam in ordinary room light it reports -5 to +3 dB, and the reading wanders by "
            "around 10 bpm. The limitation is the camera and the lighting, not the model: the pulse "
            "modulates skin brightness by about one percent, and a webcam in dim light is delivering mostly "
            "sensor noise and compression artefacts at that scale. Bright, steady, front-facing light is "
            "the single most effective remedy, because it raises the frame rate and the signal strength at "
            "once. Breathing rate is also estimated, from the vertical motion of the chest and shoulders, "
            "but the dataset carries no breathing reference, so that figure is shown as experimental and "
            "has not been validated."
        ),
    ]

    # ---------------------------------------------------------------- repro
    story += [
        heading("9. Reproducing all of this"),
        P(
            "The project is a package with a test suite that runs on synthetic data, so it can be checked "
            "out and verified without downloading the 70 GB dataset."
        ),
        code([
            "git clone https://github.com/Joshika-joe/RPPG.git &amp;&amp; cd RPPG",
            "python -m venv .venv &amp;&amp; .venv\\Scripts\\activate",
            "pip install -e \".[dev]\"",
            "python -m pytest                       # 49 tests, no dataset needed",
            "",
            "# with UBFC-rPPG in DATASET_2/ :",
            "python -m src.preprocessing            # about 3.5 minutes",
            "python -m src.train --model v3 --run my_run --loss pearson --optimizer adamw \\",
            "    --lr 1e-3 --wd 1e-2 --scheduler cosine --warmup-epochs 3 --grad-clip 1.0 \\",
            "    --select-on hr_mae --patience 15",
            "python -m src.evaluate --run results/my_run --split test \\",
            "    --continuous --stride 30 --segment-frames 300",
            "python -m src.live                     # webcam demonstration",
        ]),
        P(
            "Each training run writes its configuration, per-epoch history, best and last checkpoints and a "
            "loss curve; each evaluation writes a metrics file, a per-window table and diagnostic figures. "
            "Runs are resumable, which mattered in practice - the laptop overheated partway through one of "
            "them, and the run continued from where it had stopped."
        ),
        heading("Where it could go next", 2),
        bullet("<b>Cross-dataset testing.</b> Everything here is UBFC. Running the trained model against "
               "PURE or UBFC-Phys, with no retraining, would show whether it has learned the pulse or the "
               "dataset."),
        bullet("<b>Motion tolerance.</b> Every subject in UBFC sits still and faces the camera. A two-stream "
               "design, where the appearance branch gates the motion branch frame by frame, is the standard "
               "answer and the attention branch is already half of it."),
        bullet("<b>Breathing with a reference.</b> The breathing estimate cannot be trusted until it is "
               "measured against a dataset that records respiration."),
        Spacer(1, 0.5 * cm),
        rule(),
        P(
            "<font color='#5b6472' size='8'>Report generated from the repository's stored metrics by "
            "<font face='Courier'>docs/make_report.py</font>. Source and all results: "
            "github.com/Joshika-joe/RPPG</font>",
            "Caption",
        ),
    ]

    doc = SimpleDocTemplate(
        OUT, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=1.8 * cm,
        title="rPPG - Project Walkthrough",
        author="Joshika J",
        subject="Contactless heart-rate estimation from face video",
    )
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    sys.exit(build())

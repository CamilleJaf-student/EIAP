# Estimating Fatigue from Facial Video during Incremental Cycling

Part B group project — Computer Vision (Courses 1–3), Master 2 Health, Sport & AI.

**Design.** Two participants, each with two maximal incremental cycling tests on the same protocol (rest ≈ 2 min,
warm-up 3 min, 1-min stages to exhaustion, recovery): **v1** with the gas-exchange mask (VT1, VT2, V̇O₂max; upper
face only) and **v2** without mask (whole face). Three of the four tests were filmed. ECG and Borg CR10 in every
test. `project.ipynb` detects the face (Viola–Jones), stabilises it, extracts features in regions of interest,
aggregates them per second and tests whether they change with effort and track Borg CR10, heart rate and the
intensity domains.

**Fatigue** = the acute exertion of the test: cardiovascular strain (ECG heart rate, intensity domain) and perceived
exertion (Borg CR10). **Primary outcome (pre-specified):** Spearman correlation between mouth-opening area (mean of the
last 30 s of each stage) and Borg CR10 across the stages of each test without mask, pooled over the participants with
a repeated-measures correlation.

## Contents

```
project.ipynb            notebook: code, figures, statistics, discussion (deliverable 1)
project.py               the same notebook as plain text (jupytext), convenient for diffs
README.md                this file (deliverable 4)
requirements.txt
data/
  tests_template.xlsx        the only file to fill in  -> save as data/tests.xlsx
  raw/                       videos and ECG files (NOT shared)
outputs/                 created by the notebook
  features.csv               one row per second, one column per feature, condition labels (deliverable 2)
  features_second_stats.csv  mean / SD / min / max of every per-frame feature for each second
  params.json, tables/, figures/
docs/reference_protocols.md  acquisition and exercise protocols of the three reference articles
tools/
  make_synthetic_session.py  synthetic test data (schematic face + raw ECG) to test the pipeline
  frames_sheet.py            contact sheet of frames with timestamps (sync taps, gray card, face box)
  anonymize_clip.py          pixelated video excerpt for the slides
  run_notebook.py            rebuild project.ipynb from project.py and execute it
outputs_demo/            results of the synthetic demo (data_demo/ is created on the first DEMO run)
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook project.ipynb
```

OpenCV must be a **4.x** wheel: the Haar cascade XML files are not shipped with OpenCV 5 wheels (alternatively, copy
`haarcascade_frontalface_default.xml` and `haarcascade_eye.xml` into `./cascades/`).

## Test the pipeline before filming

`DEMO = True` (first code cell) runs everything on two synthetic tests: a schematic animated face with known heart
rate, breathing rate, mouth opening, redness, sweat and head bob, plus a raw ECG with sync taps. The data are created
on the first run (15–20 min), then processed (15–20 min once, about a minute afterwards thanks to the cache). The
synthetic face only checks that the code runs and recovers known signals; it says nothing about real faces, and no
mask is drawn on it (the `mask` flag of the first demo test only exercises the code path).

## Recording protocol

**Exercise (every test).** Seated, constant cadence ≥ 75 rpm (pedalling then stays above the breathing band). Rest
≈ 2 min on the bike → warm-up 3 min → 1-min stages of increasing power to exhaustion → recovery on the bike, still
filmed. Same stages in both tests. The subject does **not talk**; the partner announces each stage aloud (the audio
gives the time stamps) and notes the power. Same fan setting (ideally none) in both tests.

**Borg CR10.** Printed scale next to the camera at face height, pointed at during the last 10 s of every stage, at the
end of rest and at the end of the filmed recovery.

**Mask (v1).** Check before the real test that the face is still detected with the mask on: run the notebook on a
30-s clip. If Viola–Jones fails, give a fixed `face_box` in `tests.xlsx` (the tripod and the ergometer do not move).

**iPhone.** Settings › Camera › Formats: *Most Compatible* (H.264). Record Video: 1080p at 30 fps (or 25 fps with
the PAL format if bands or pulsing appear under the room lights), **HDR Video off**, *Auto FPS* off, *Enhanced
Stabilization* off, *Lock Camera* on, **Lock White Balance on**. Rear main camera (1×), standard Video mode,
landscape, tripod, ≈ 0.8 m, lens at eye height, face filling 40–60 % of the frame height. Airplane mode (a call
stops the recording). Long-press on the face → *AE/AF Lock*, then do not touch the screen. One continuous clip per
test, from the sync taps to the end of recovery; transfer the original file by cable or AirDrop (messaging apps
re-compress) and do not trim it. A matte
**gray card** in the frame, at face distance, under the same light (`ref_roi`) corrects exposure and white-balance
drift. Same camera position, light and settings in both tests. No glasses, make-up or cap; hair tied.

**ECG.** Raw CSV with one or several leads (a 12-lead export is fine: the lead with the fewest rejected beats is
kept). **Synchronisation:** leave `ecg_offset_s` empty and the highest heart rate is aligned with the end of effort
read in the video (± 5 s). §4 prints an independent check (pedalling noise of the ECG against head motion of the
video) and §5 overlays the two curves. For a tighter synchronisation in a future recording, tap an electrode three
times in front of the camera: §4 prints the tap times in ECG time; read the first tap in the video with
`python tools/frames_sheet.py data/raw/S1_v2_video.mov --t0 2 --t1 12 --step 0.2` and set
`ecg_offset_s = t_tap_video − t_tap_ecg`.

## Preparing your data

The videos are **not cut**: one whole file per test, and one workbook that says where the segments are.

1. Copy the videos and the ECG files into `data/raw/` (e.g. `S1_v2_video.mov`, `S1_v2_ecg.csv`; use pseudonyms, the
   names end up in `features.csv` and in the figures).
2. Save `data/tests_template.xlsx` as `data/tests.xlsx` and fill in the yellow cells (legend in its first sheet):

   **Sheet `tests`** — one row per test.

   | Column | Meaning |
   |---|---|
   | `session_id`, `subject_id` | e.g. `S1_v2`, `S1` |
   | `video_file`, `ecg_file` | relative to `data/`; empty for a test that was not filmed |
   | `mask` | 1 for the test with the gas-exchange mask: lower-face features are discarded |
   | `t_rest_start`, `t_warmup_start` | video times as `mm:ss` (or seconds); empty if not in the video |
   | `first_stage`, `t_first_stage` | number and start time of the first stage whose start is in the video (1 in general) |
   | `stage_s` | stage duration (60); every stage is derived from `t_first_stage` and `stage_s` |
   | `t_effort_end`, `t_recovery_end` | end of exercise (the last stage may be incomplete: analysed if ≥ 15 s) and of the filmed recovery |
   | `ecg_offset_s` | empty: highest heart rate aligned with the end of effort; or a value, t_video = t_ecg + offset |
   | `vt1_W`, `vt2_W` | power at VT1 and VT2, typed once per subject |
   | `ecg_fs`, `ref_roi`, `face_box` | optional: ECG sampling rate if the file has no time column; gray card and fixed face box as `x,y,w,h` fractions of the upright frame |
   | `rotate` | optional: clockwise rotation of the video (0 / 90 / 180 / 270); empty = found automatically |

   **Sheet `paliers`** — one row per segment and test: `session_id`, `palier` (`REPOS`, `WARMUP`, `P1` … `Pn`,
   `RECUP`), `power_W`, `borg_cr10` and, for the masked test, the stage means `vo2`, `ve`, `bf` of the metabolic cart.

   `python tools/frames_sheet.py VIDEO --t0 10 --t1 10 --grid` draws a 10 % grid to read `ref_roi` and `face_box`.
3. Set `DEMO = False` and *Run All*. The notebook prints `CHECK …` if the times and the stage rows disagree. Check
   §5 (ECG, ROI placement — set `ANONYMIZE_FIGURES = False` locally, detection rate) before reading any statistics.
   If the ROIs are misplaced, adjust `PARAMS["roi"]` once, for every test.

Portrait videos and videos stored sideways are handled: the notebook keeps the orientation in which Viola–Jones finds
the face (printed in §3; `rotate` overrides it, and `tools/frames_sheet.py --rotate 90` shows the frames the same
way). The smallest face searched is 20 % of the shorter side of the frame.

A video that starts in the middle of the protocol is handled (`first_stage` > 1, rest left empty): its stages keep
their true numbers and the comparisons with rest are skipped for that test. A test without ECG file has no heart rate.

Analysis windows (same for every test): rest = last 120 s of rest, effort = last 60 s of exercise, recovery = first
120 s after exercise, stage = last 30 s of each stage. The warm-up is labelled but not analysed.

## `features.csv` columns

| Column | Meaning |
|---|---|
| `session_id`, `subject_id`, `t_s` | test, pseudonymous subject, second of video time |
| `segment`, `condition`, `stage`, `power_W`, `borg_cr10` | labels from `tests.xlsx` (condition: rest / warmup / effort / recovery) |
| `window` | analysis window (rest / effort / recovery; empty elsewhere) |
| `stage_window` | True for the last 30 s of each stage (and the rest window) |
| `t_since_end` | seconds since the end of exercise |
| `hr_ecg`, `hr_pct_max` | ECG heart rate (bpm) and % of the maximum measured in the tests |
| `mouth_area` … `blink_rate` | facial features, per-second mean (units in the notebook, §3 and §6) |
| `eyes_n`, `sharpness`, `skin_frac`, `det_rate`, `n_frames` | quality-control variables |

## Reproducibility

One `PARAMS` dictionary for every sequence, saved to `outputs/params.json`; per-frame features cached under a hash of
the parameters and of the video file; fixed seed (42) for bootstrap and permutations; every figure and table written by
the notebook; library versions printed in the last cell. `python tools/run_notebook.py` re-executes everything.

## Data availability and anonymisation

Raw videos and ECG recordings identify the participant and are **not shared**. Shared: `features.csv` (numbers only),
tables, figures with the face pixelated (`ANONYMIZE_FIGURES = True`) and, for the slides, clips pixelated with
`tools/anonymize_clip.py` (watch them frame by frame before sharing). The participant consented to be filmed.

## Main references

* Timme S, Brand R. Affect and exertion during incremental physical exercise: examining changes using automated facial
  action analysis and experiential self-report. *PLoS ONE* 2020;15(2):e0228739.
* Khanal SR, Sampaio J, Exel J, Barroso J, Filipe V. Using computer vision to track facial color changes and predict
  heart rate. *J Imaging* 2022;8(9):245.
* Khanal SR, Paulino D, Sampaio J, Barroso J, Reis A, Filipe V. A review on computer vision technology for physical
  exercise monitoring. *Algorithms* 2022;15(12):444.
* Khanal SR, Fonseca A, Marques A, Barroso J, Filipe V. Physical exercise intensity monitoring through eye-blink and
  mouth's shape analysis. TISHW 2018.
* Miles KH, Clark B, Périard JD, Goecke R, Thompson KG. Facial feature tracking: a psychophysiological measure to assess
  exercise intensity? *J Sports Sci* 2018;36(8):934–941.
* Enoka RM, Duchateau J. Translating fatigue to human performance. *Med Sci Sports Exerc* 2016;48(11):2228–2238.
* Viola P, Jones M. Rapid object detection using a boosted cascade of simple features. CVPR 2001.
* Pan J, Tompkins WJ. A real-time QRS detection algorithm. *IEEE Trans Biomed Eng* 1985;32(3):230–236.
* Bakdash JZ, Marusich LR. Repeated measures correlation. *Front Psychol* 2017;8:456.
* Chai D, Ngan KN. Face segmentation using skin-color map in videophone applications. *IEEE Trans Circuits Syst Video
  Technol* 1999;9(4):551–564.

# Acquisition and exercise protocols of the reference studies

Extracted from the three articles of the project, for later comparison with our own protocol. "n.r." = not reported
in the article. Page numbers refer to the PDF of each article.

## Side-by-side

| | Timme & Brand 2020 (PLoS ONE) | Khanal et al. 2022 (J Imaging) | Our protocol |
|---|---|---|---|
| **Participants** | 132 tested, **113 analysed** (19 excluded: 6 recording failures, 6 with > 10 % of frames without face detection, 7 external disturbances); sport-science students, 18–36 y (21.6 ± 2.9), 53 women; **no beard, no spectacles** | **9** students (6 men, 3 women), 26.9 ± 6.0 y, all white Caucasian; protocol familiarisation | 1 subject, [age, sex] |
| **Ergometer** | Indoor bike ergometer (model n.r.) | Wattbike cycle ergometer | [Model], power from the bike computer |
| **Exercise** | **Maximal**: 25 W, **+25 W every 2 min** until voluntary exhaustion (required power no longer produced); ceiling 300 W for 2 min (≤ 26 min); 5–13 stages per subject; 5-min cool-down (easy cycling) | **Sub-maximal ramp**: 5-min warm-up, then 75 W, **+15 W/min** until 85 % of predicted HRmax (208 − 0.7 × age) or cadence no longer maintained; duration 8–16 min | **Two maximal tests, 30 min apart**, identical rest, 1-min stages (+[25] W) and recovery; ≥ 5 min of passive recovery filmed |
| **Warm-up / rest** | No warm-up reported; 1-min HR baseline in the 54 subjects with ECG | 5-min warm-up; no resting recording | Rest 3 min seated on the bike (last 2 min analysed) |
| **Perceived exertion** | **Borg RPE 6–20** and Feeling Scale (−5 to +5), every 2 min at the end of each stage; scales shown 10 s on a 22" monitor at 1.5 m; **answered verbally** — these periods were **cut from the video** | None | **Borg CR10**, pointed at on a printed scale at the end of each stage (no speech) |
| **Physiological reference** | HR in 54 of the subjects: Shimmer3 ECG, 512 Hz (plausibility check only). **No gas exchange** (stated as a limitation; proposed as future work, with VT and RCP) | HR: Polar T31, 1 Hz, synchronised with the Wattbike (power at 100 Hz). No RPE, no gas exchange | Raw ECG in both tests; **gas exchange in Test 1** (VT1, VT2, V̇O₂max) |
| **Camera** | Logitech HD Pro C920 webcam, **30 fps** (resolution n.r.) | Video camera (model n.r.), **25 fps, 1080 × 1920 px** | iPhone 15, 1080p, 30 fps, H.264, HDR off |
| **Position** | On top of the ergometer screen, **0.4 m** from the face, **20° from below** | Tripod, **≈ 1 m**, frontal plane, 90° to the face | Tripod, ≈ 0.8 m, face height, frontal |
| **Lighting** | "Evenly and clearly lit laboratory" | Fluorescent tubes (50 Hz mains) | Diffuse frontal light; exposure locked; gray card in frame |
| **Instructions** | No verbal encouragement or feedback; experimenter out of sight | **No talking**; look straight at the lens; free facial expression | No talking; partner announces the stages |
| **Face detection** | Affectiva Affdex in iMotions 7.2 (**Viola–Jones** inside), 34 landmarks | **Viola–Jones**; face cropped and resized to 400 × 400 px | Viola–Jones (OpenCV Haar cascade), median of 9 boxes |
| **Pre-processing** | — | Wiener–Hunt deblurring of blurred frames (variance threshold), Gaussian filter; **chromatic normalisation** c / (R + G + B) | Blurred frames excluded (variance of the Laplacian); gray-card gain |
| **Feature** | 20 facial actions (probability 0–100 % per frame); score per stage = **% of the 2 min with probability ≥ 10** | Mean colour of a **16 × 16 px patch on the lower forehead** in RGB, HSV, YCbCr, Lab, YUV; 1 value per second; **median filter over 201 points** + moving average | Mouth-cavity area, a\* / H / S on forehead and cheeks, box jitter and optical flow, blinks, glabella gradient, specular pixels; mean per second |
| **Unit of analysis** | One value per 2-min stage | One value per second (heavily smoothed) | Last 30 s of each stage; 10-s blocks for condition contrasts |
| **Statistics** | Repeated-measures correlation; multilevel regression (R 3.6.0, nlme 3.1-139) | Per-subject multivariate **autoregression of lag 1 that includes HR(t − 1)** as predictor; global polynomial support-vector regression (degree 3) of % HRmax; Python 3.5 | Spearman ρ (permutation, CI), rmcorr, Kruskal–Wallis / Mann–Whitney, Cliff's δ with bootstrap CI, Wilcoxon on matched stages |
| **Main result** | *Mouth open*: r_rm = 0.70 with RPE, −0.55 with valence; *jaw drop* 0.51; *nose wrinkle* specific to valence. Final HR 174.6 ± 16.1 bpm, final RPE 19.3 ± 1.0, HR–RPE r = 0.82 | Global model, HSV: RMSE 6.75 (% HRmax), R² = 0.78; individual models better; high between-subject heterogeneity | — |

## Points that matter for the comparison

* **Stage length.** Timme & Brand use 2-min stages, Khanal a 1-min ramp. With 1-min stages our stage means are
  closer to Khanal's ramp; heart rate and ventilation lag more behind power than in Timme & Brand.
* **RPE scale and speech.** Timme & Brand asked for a verbal answer and removed those seconds from the video. We
  avoid speech altogether (pointing), so no frame has to be removed.
* **Feature definition.** Their mouth score is a *proportion of time* in which the mouth is detected as open; ours is
  a *mean area*. For a direct comparison the notebook also computes `mouth_open_pct`: the percentage of frames of
  each second with `mouth_area` above the opening threshold (0.3 % of the face area).
* **Exclusions.** Timme & Brand excluded recordings with more than 10 % of frames without a detected face; our
  notebook reports the detection rate per condition (§5) so that the same criterion can be applied.
* **Camera geometry.** 0.4 m and 20° from below (webcam on the ergometer) against ≈ 1 m and frontal (tripod). A low
  angle exposes the mouth and jaw more; our frontal view is closer to Khanal's.
* **Strength of the colour evidence.** Khanal's per-subject model predicts HR(t) from HR(t − 1) and the colour at
  t − 1: its low error mostly reflects the autocorrelation of heart rate. The global model (R² = 0.78) is the result
  that speaks about colour. The % HRmax is computed as 208 − 0.7 × age in the protocol and as 220 − age in the
  statistics section. The colour signal is smoothed over 201 s, which removes anything faster than the ramp itself.
* **Thresholds.** Neither study measured gas exchange. Timme & Brand suggest that VT could mark the transition to
  mouth breathing (more *mouth open*) and the RCP the transition to *jaw drop*, and propose respiratory gas analysis
  in a pre-test: this is the design of our Test 1 → Test 2.

## Studies summarised in the review (Khanal et al. 2022, Algorithms 15:444)

The review reports no protocol of its own (literature search with a PRISMA flow diagram). What it reports about
the studies closest to ours — details beyond these lines are in the original papers, which we have not read:

| Study | Exercise | Acquisition and method (as reported by the review) | Result |
|---|---|---|---|
| Khanal et al. 2018 (TISHW) | Sub-maximal exercise on a cycle ergometer | Eye-blink rate and mouth open–close rate tracked with Viola–Jones | Blinks identified with 96 % accuracy; more eye and mouth movement at higher intensity |
| Miles et al. 2018 (J Sports Sci 36:934) | Incremental exercise on a cycle ergometer, with power, HR, RPE, blood lactate (two lactate thresholds) and maximal aerobic power | Facial feature tracking (IntraFace, 49 facial points and head movement, according to Timme & Brand) | Reliability higher for the lower face; non-linear relation between facial movement and power; movement increased from the lactate threshold to maximal aerobic power |
| Aoki & Nakamura 2018 (Sports 6:23) | Incremental test on a cycle ergometer | Kinect depth camera; breathing rate from the periodic signal | Breathing rate measurable above 160 W; ventilatory threshold within ±10 W of the gas analyser |
| Irani et al. 2014 (ICIP) | Maximal and sub-maximal dumbbell lifting, dynamometer as reference | Tracking of facial feature points; vibration by power-spectral analysis | Vibration of the points increases towards fatigue |
| Uchida et al. 2018 (PLoS ONE 13:e0208834) | Resistance training at several loads | FACS coding and facial surface EMG | Mild but significant association |
| Lopez et al. (thermal imaging + deep learning) | Exercise-induced fatigue, rested vs fatigued | RGB, near-infrared and thermal images; pre-trained CNNs per facial region | Best classification from the region around the mouth |

## Code and data availability

| Article | Supplementary material | Data | Code |
|---|---|---|---|
| Timme & Brand 2020 | None in the article | Open data on OSF: https://osf.io/z8rv7/ | Not stated. Facial coding was done with commercial software (Affectiva Affdex in iMotions 7.2); statistics in R (nlme). Check the OSF page for R scripts |
| Khanal et al. 2022 (J Imaging) | None | "Not applicable" | Not shared ("dedicated codes written in Python 3.5") |
| Khanal et al. 2022 (review) | PRISMA-P checklist only | "Not applicable" | None |

"""
Synthetic test data for the facial-exertion pipeline.

Generates two simulated maximal incremental cycling tests of one subject (rest -> 1-min stages ->
passive recovery): a 720p video of a schematic face, a raw ECG CSV with three sync taps, the
ground-truth signals used to animate the face, and the metadata files of the notebook
(sessions.csv, segments.csv, cpet_stages.csv). Test 1 is flagged as performed with the
gas-exchange mask (no mask is drawn: the flag only exercises the code path).

Purpose: check the whole pipeline end-to-end BEFORE real data exist (does the code
run, does it recover a known heart rate, breathing rate, mouth and colour trend?).
It is NOT a model of a real face and says nothing about what real data will show.

Usage (from the project root):
    python tools/make_synthetic_session.py --out data_demo --seed 1
    python tools/make_synthetic_session.py --out data_demo --metadata-only
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import pandas as pd

W, H = 1280, 720


# ----------------------------------------------------------------------------- physiology
def first_order(target: np.ndarray, dt: float, tau_on: float, tau_off: float) -> np.ndarray:
    """First-order kinetics towards a moving target (different on/off time constants)."""
    y = np.empty_like(target)
    y[0] = target[0]
    for i in range(1, len(target)):
        tau = tau_on if target[i] > y[i - 1] else tau_off
        y[i] = y[i - 1] + (target[i] - y[i - 1]) * (1 - np.exp(-dt / tau))
    return y


def simulate_physiology(subj: dict, fps: float, rng: np.random.Generator) -> pd.DataFrame:
    rest_s, n_st, st_s, rec_s = subj["rest_s"], subj["n_stages"], subj["stage_s"], subj["recovery_s"]
    dur = rest_s + n_st * st_s + rec_s
    t = np.arange(0, dur, 1 / fps)
    dt = 1 / fps
    ex_end = rest_s + n_st * st_s
    stage = np.where((t >= rest_s) & (t < ex_end), ((t - rest_s) // st_s + 1), 0).astype(int)
    intensity = stage / n_st                      # 0..1, step-wise
    pedalling = (stage > 0).astype(float)
    rec = t >= ex_end

    hr_t = subj["hr_rest"] + (subj["hr_peak"] - subj["hr_rest"]) * intensity ** 1.1
    hr_t[rec] = subj["hr_rest"] + 12
    hr = first_order(hr_t, dt, 22.0, 45.0)

    br_t = 13 + 32 * intensity ** 1.6
    br_t[rec] = 14
    br = first_order(br_t, dt, 20.0, 50.0)       # ventilation stays high in early recovery

    vent = first_order(intensity ** 1.3, dt, 20.0, 50.0)
    mouth_mean = 0.04 + 0.55 * vent
    mouth_amp = 0.03 + 0.30 * vent

    red_t = intensity.copy()
    red_t[rec] = 0.0
    redness = first_order(red_t, dt, 120.0, 45.0)  # slow on, faster off (non-thermal reflexes)

    sweat = np.zeros_like(t)
    for i in range(1, len(t)):
        drive = max(intensity[i] - 0.35, 0.0) * 0.012
        sweat[i] = min(1.0, sweat[i - 1] + drive * dt - (0.0 if drive > 0 else sweat[i - 1] * dt / 400))

    frown_t = np.clip((intensity - 0.6) / 0.4, 0, 1)
    frown_t[rec] = 0
    frown = first_order(frown_t, dt, 8.0, 15.0)

    # phases (integrals) for oscillations
    br_phase = 2 * np.pi * np.cumsum(br / 60) * dt
    hr_phase = 2 * np.pi * np.cumsum(hr / 60) * dt
    cad_hz = subj["cadence_rpm"] / 60
    cad_phase = 2 * np.pi * cad_hz * t + rng.uniform(0, 2 * np.pi)

    mouth = np.clip(mouth_mean + mouth_amp * np.sin(br_phase), 0, 1)
    bob_amp = pedalling * (1.0 + 5.0 * intensity)
    sway_amp = pedalling * (0.5 + 3.0 * intensity)
    drift_x = np.cumsum(rng.normal(0, 0.03, len(t)))
    drift_x -= pd.Series(drift_x).rolling(int(10 * fps), min_periods=1, center=True).mean().values
    dy = bob_amp * np.sin(2 * cad_phase) + 0.4 * np.sin(br_phase)
    dx = sway_amp * np.sin(cad_phase) + drift_x

    blink_rate = (12 + 12 * intensity) / 60     # per second
    blink = np.zeros(len(t), bool)
    i = 0
    n_bl = max(2, int(round(0.13 * fps)))
    while i < len(t):
        if rng.random() < blink_rate[i] * dt:
            blink[i:i + n_bl] = True
            i += n_bl + int(0.5 * fps)
        else:
            i += 1

    gain = 1 + 0.03 * np.sin(2 * np.pi * t / 240)   # slow exposure drift (gray card can fix it)
    pulse = np.sin(hr_phase)

    borg = np.zeros(len(t))
    return pd.DataFrame(dict(t=t, stage=stage, intensity=intensity, hr=hr, br=br, mouth=mouth,
                             redness=redness, sweat=sweat, frown=frown, dx=dx, dy=dy,
                             blink=blink, gain=gain, pulse=pulse, pedalling=pedalling,
                             borg=borg, hr_phase=hr_phase))


# ----------------------------------------------------------------------------- rendering
class FaceRenderer:
    def __init__(self, subj: dict, rng: np.random.Generator):
        self.skin0 = np.array(subj["skin_bgr"], float)
        self.cx, self.cy = 640, 360
        self.fw, self.fh = 150, 195
        g = np.linspace(150, 118, W)
        bg = np.stack([g + 4, g, g - 4], -1)[None, :, :].repeat(H, 0)
        # gray card (reference ROI for colour normalisation)
        bg[300:420, 90:210] = 128
        self.bg = bg.astype(np.float32)
        self.bg_u8 = np.clip(bg, 0, 255).astype(np.uint8)
        yy, xx = np.mgrid[0:H, 0:W]
        r2 = ((xx - self.cx + 60) / 260.0) ** 2 + ((yy - self.cy + 80) / 320.0) ** 2
        self.shade = (1.05 - 0.18 * np.clip(r2, 0, 1.5)).astype(np.float32)[..., None]
        # candidate sweat highlight spots (forehead + cheeks), fixed per subject
        spots = []
        while len(spots) < 60:
            x = rng.uniform(-110, 110)
            y = rng.uniform(-150, 90)
            in_forehead = -150 < y < -95 and abs(x) < 85
            in_cheek = 15 < y < 85 and 45 < abs(x) < 115
            if in_forehead or in_cheek:
                spots.append((x, y, rng.uniform(1.5, 3.5)))
        self.spots = spots
        # sensor noise: bank of noise fields + random offsets so that the noise never repeats periodically
        self.noise = [rng.normal(0, 2.2, (H + 16, W + 16, 3)).astype(np.float32) for _ in range(8)]
        self.rng = rng
        # relative pulsatile amplitude per channel (B, G, R): green strongest, red weakest (blood absorption)
        self.ppg_amp = np.array([-0.0040, -0.0060, -0.0025], np.float32)
        self.k = 0

    def render(self, row) -> np.ndarray:
        img = np.zeros((H, W, 3), np.uint8)
        ox, oy = self.cx + int(round(row.dx)), self.cy + int(round(row.dy))
        red = row.redness
        skin = self.skin0 + np.array([-6, -14, 9]) * red
        sk = tuple(int(np.clip(v, 0, 255)) for v in skin)
        fw, fh = self.fw, self.fh
        hair = (30, 35, 45)
        cv2.rectangle(img, (ox - 70, oy + 150), (ox + 70, H), tuple(int(v * 0.9) for v in sk), -1)
        cv2.ellipse(img, (ox, H + 60), (300, 200), 0, 0, 360, (90, 60, 40), -1)
        cv2.ellipse(img, (ox, oy - 40), (fw + 18, fh + 5), 0, 180, 360, hair, -1)
        cv2.ellipse(img, (ox, oy), (fw, fh), 0, 0, 360, sk, -1)
        cv2.ellipse(img, (ox, oy - fh + 40), (fw + 5, 70), 0, 180, 360, hair, -1)
        brow_y = oy - 55 + int(round(4 * row.frown))
        for s in (-1, 1):
            a0, a1 = (200, 360) if s < 0 else (180, 340)
            cv2.ellipse(img, (ox + s * 60, brow_y), (40, 12), 0, a0, a1, (40, 45, 60), 8)
        if row.frown > 0.02:
            c = tuple(int(v * (1 - 0.35 * row.frown)) for v in sk)
            for s in (-1, 1):
                cv2.line(img, (ox + s * 8, oy - 62), (ox + s * 6, oy - 38), c, 3)
        for s in (-1, 1):
            ex, ey = ox + s * 60, oy - 25
            if row.blink:
                cv2.line(img, (ex - 28, ey), (ex + 28, ey), (50, 60, 80), 4)
            else:
                cv2.ellipse(img, (ex, ey), (28, 14), 0, 0, 360, (235, 235, 240), -1)
                cv2.circle(img, (ex, ey), 11, (50, 40, 30), -1)
                cv2.circle(img, (ex, ey), 5, (10, 10, 10), -1)
                cv2.ellipse(img, (ex, ey), (28, 14), 0, 180, 360, (40, 40, 50), 3)
        nose = tuple(int(v * 0.75) for v in sk)
        cv2.line(img, (ox - 5, oy - 20), (ox - 12, oy + 45), nose, 4)
        cv2.ellipse(img, (ox, oy + 50), (25, 10), 0, 0, 180, tuple(int(v * 0.7) for v in sk), 3)
        mh = int(round(3 + 32 * row.mouth))
        cv2.ellipse(img, (ox, oy + 105), (42, mh), 0, 0, 360, (70, 70, 150), -1)
        if mh > 6:
            cv2.ellipse(img, (ox, oy + 105), (36, mh - 6), 0, 0, 360, (25, 20, 40), -1)
        n_sp = int(round(len(self.spots) * row.sweat))
        for (x, y, r) in self.spots[:n_sp]:
            cv2.circle(img, (ox + int(x), oy + int(y)), int(round(r)), (248, 248, 248), -1)

        # composite only the region around the head (speed); background stays static
        x0, x1, y0, y1 = self.cx - 260, self.cx + 260, self.cy - 260, H
        crop = img[y0:y1, x0:x1]
        mask = crop.any(-1, keepdims=True)
        skin_px = np.all(crop == np.array(sk, np.uint8), axis=-1).astype(np.float32)
        f = np.where(mask, crop.astype(np.float32), self.bg[y0:y1, x0:x1])
        f = f * (1.0 + skin_px[..., None] * self.ppg_amp[None, None, :] * float(row.pulse))   # blood volume pulse
        f = cv2.GaussianBlur(f, (0, 0), 1.6)
        f = f * self.shade[y0:y1, x0:x1] * row.gain
        nz = self.noise[int(self.rng.integers(len(self.noise)))]
        oxn, oyn = int(self.rng.integers(16)), int(self.rng.integers(16))
        f += nz[y0 + oyn:y1 + oyn, x0 + oxn:x1 + oxn]
        self.k += 1
        if row.pedalling > 0 and row.intensity > 0.6:
            kk = 1 + int(round(4 * (row.intensity - 0.6) / 0.4))
            if kk > 1:
                f = cv2.blur(f, (1, kk))
        out = self.bg_u8.copy()
        out[y0:y1, x0:x1] = np.clip(f, 0, 255).astype(np.uint8)
        card = self.bg[300:420, 90:210] * row.gain + self.noise[0][300:420, 90:210]
        out[300:420, 90:210] = np.clip(card, 0, 255).astype(np.uint8)   # gray card sees the same exposure drift
        return out


# ----------------------------------------------------------------------------- ECG
def synth_ecg(phys: pd.DataFrame, fs: float, offset_s: float, taps_video_s: list[float],
              rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """Raw ECG (mV) whose clock starts `-offset_s` s relative to the video: t_video = t_ecg + offset_s."""
    t_v = phys.t.values
    hr_v = phys.hr.values
    t_ecg_start = 0.0
    dur_ecg = t_v[-1] - offset_s + 2.0
    t = np.arange(0, dur_ecg, 1 / fs)
    tv = t + offset_s
    hr = np.interp(tv, t_v, hr_v)
    inten = np.interp(tv, t_v, phys.intensity.values)
    # beat times (with a little HRV)
    beats = []
    tb = 0.3
    while tb < t[-1] - 0.5:
        beats.append(tb)
        h = np.interp(tb + offset_s, t_v, hr_v)
        rr = 60 / h * (1 + rng.normal(0, 0.012 * (1 - np.interp(tb + offset_s, t_v, phys.intensity.values))))
        tb += rr
    beats = np.array(beats)
    x = np.zeros_like(t)
    comps = [(-0.20, 0.15, 0.025), (-0.03, -0.10, 0.010), (0.0, 1.20, 0.011), (0.03, -0.25, 0.011)]
    for b in beats:
        i0 = int((b - 0.45) * fs)
        i1 = int((b + 0.6) * fs)
        i0, i1 = max(i0, 0), min(i1, len(t))
        tt = t[i0:i1] - b
        rr_loc = 60 / np.interp(b + offset_s, t_v, hr_v)
        seg = sum(a * np.exp(-0.5 * ((tt - mu) / s) ** 2) for mu, a, s in comps)
        seg += 0.30 * np.exp(-0.5 * ((tt - 0.25 * np.sqrt(rr_loc)) / 0.05) ** 2)
        x[i0:i1] += seg
    x += 0.25 * np.sin(2 * np.pi * 0.25 * t + 1.0)                    # baseline wander
    x += rng.normal(0, 0.02 + 0.06 * inten, len(t))                     # EMG/noise grows with effort
    for tv_tap in taps_video_s:                                        # sync taps (big artefacts)
        c = tv_tap - offset_s
        x += 3.0 * np.exp(-0.5 * ((t - c) / 0.015) ** 2) - 2.0 * np.exp(-0.5 * ((t - c - 0.04) / 0.02) ** 2)
    return pd.DataFrame(dict(t=t, ecg_mV=x)), beats


# ----------------------------------------------------------------------------- main
SUBJECTS = {   # two tests of the same subject; the second one starts pre-fatigued (higher resting HR, one stage less)
    "S01": dict(subject_id="A", mask=1, skin_bgr=(130, 160, 205), hr_rest=66, hr_peak=184, rest_s=150,
                n_stages=9, stage_s=60, recovery_s=330, cadence_rpm=80, fps=30.0,
                ecg_fs=250, ecg_offset_s=-4.0, ecg_style="time_ms"),
    "S02": dict(subject_id="A", mask=0, skin_bgr=(112, 142, 192), hr_rest=72, hr_peak=176, rest_s=150,
                n_stages=8, stage_s=60, recovery_s=330, cadence_rpm=85, fps=25.0,
                ecg_fs=500, ecg_offset_s=2.5, ecg_style="no_time"),
}
VT1_W, VT2_W, WEIGHT_KG = 125, 200, 70


def render_session(sid: str, out_dir: str, seed: int) -> None:
    """Write the video, the raw ECG and the ground truth of one test."""
    subj = SUBJECTS[sid]
    rng = np.random.default_rng(seed + int(sid[1:]))
    phys = simulate_physiology(subj, subj["fps"], rng)
    raw = os.path.join(out_dir, "raw")
    os.makedirs(raw, exist_ok=True)
    vw = cv2.VideoWriter(os.path.join(raw, f"{sid}_video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), subj["fps"], (W, H))
    rend = FaceRenderer(subj, rng)
    for row in phys.itertuples():
        vw.write(rend.render(row))
    vw.release()
    ecg, _ = synth_ecg(phys, subj["ecg_fs"], subj["ecg_offset_s"], [6.0, 6.6, 7.2], rng)
    epath = os.path.join(raw, f"{sid}_ecg.csv")
    if subj["ecg_style"] == "time_ms":
        pd.DataFrame({"timestamp_ms": np.round(ecg.t * 1000, 3), "ECG_uV": np.round(ecg.ecg_mV * 1000, 1)}
                     ).to_csv(epath, index=False, sep=";")
    else:
        pd.DataFrame({"ECG": np.round(ecg.ecg_mV, 4)}).to_csv(epath, index=False)
    phys["sec"] = phys.t.astype(int)
    phys.groupby("sec").agg(hr=("hr", "mean"), br=("br", "mean"), mouth=("mouth", "mean"), redness=("redness", "mean"),
                            sweat=("sweat", "mean"), frown=("frown", "mean"),
                            blinks=("blink", lambda b: int((np.diff(b.astype(int)) == 1).sum()))
                            ).to_csv(os.path.join(out_dir, f"truth_{sid}.csv"))


def write_metadata(out_dir: str, seed: int) -> None:
    """sessions.csv, segments.csv and cpet_stages.csv (independent of the rendering random stream)."""
    rng = np.random.default_rng(seed + 100)
    sess, segs = [], []
    for sid, subj in SUBJECTS.items():
        rest_s, st_s, n = subj["rest_s"], subj["stage_s"], subj["n_stages"]
        ex_end = rest_s + n * st_s
        borg = [float(np.clip(np.round((0.5 + 9.5 * (k / n) ** 1.4 + rng.normal(0, 0.4)) * 2) / 2, 0, 10))
                for k in range(1, n + 1)]
        rows = [dict(segment="rest", condition="rest", stage=0, t_start_s=rest_s - 120, t_end_s=rest_s, power_W=0,
                     borg_cr10=0.5)]
        rows += [dict(segment=f"stage{k:02d}", condition="effort", stage=k, t_start_s=rest_s + (k - 1) * st_s,
                      t_end_s=rest_s + k * st_s, power_W=50 + 25 * (k - 1), borg_cr10=borg[k - 1]) for k in range(1, n + 1)]
        rows.append(dict(segment="recovery", condition="recovery", stage=-1, t_start_s=ex_end, t_end_s=ex_end + 300,
                         power_W=0, borg_cr10=1.0))
        segs.append(pd.DataFrame(rows).assign(session_id=sid))
        sess.append(dict(session_id=sid, subject_id=subj["subject_id"], video_path=f"raw/{sid}_video.mp4",
                         ecg_path=f"raw/{sid}_ecg.csv", ecg_fs=subj["ecg_fs"] if subj["ecg_style"] == "no_time" else "",
                         ecg_offset_s=subj["ecg_offset_s"], ref_roi="0.0703,0.4167,0.0938,0.1667", mask=subj["mask"],
                         face_box="", vt1_W=VT1_W, vt2_W=VT2_W, notes="synthetic demo"))
    pd.DataFrame(sess).to_csv(os.path.join(out_dir, "sessions.csv"), index=False)
    pd.concat(segs).to_csv(os.path.join(out_dir, "segments.csv"), index=False)
    # stage means of the metabolic cart in Test 1 (S01): V'O2 (ml/min/kg), V'E (l/min), breathing frequency (/min)
    s1 = SUBJECTS["S01"]
    phys = simulate_physiology(s1, 5.0, np.random.default_rng(seed + 1))
    rows = []
    for k in range(0, s1["n_stages"] + 1):
        t1 = s1["rest_s"] + k * s1["stage_s"]
        power = 0 if k == 0 else 50 + 25 * (k - 1)
        bf = phys[(phys.t >= t1 - 30) & (phys.t < t1)].br.mean()
        vo2 = (10.8 * power + 7 * WEIGHT_KG) / WEIGHT_KG if k else 4.5
        ve = 9 + 0.21 * power + 0.5 * max(0, power - VT1_W) ** 1.05 * 0.3 + 0.9 * max(0, power - VT2_W)
        rows.append(dict(subject_id="A", stage=k, vo2=round(vo2, 1), ve=round(ve, 1), bf=round(bf, 1)))
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "cpet_stages.csv"), index=False)


def make_all(out_dir: str, seed: int = 1, render: bool = True) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if render:
        for sid in SUBJECTS:
            print("rendering synthetic test", sid, "(several minutes)...")
            render_session(sid, out_dir, seed)
    write_metadata(out_dir, seed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_demo")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--metadata-only", action="store_true", help="rewrite the CSV metadata without rendering the videos")
    a = ap.parse_args()
    make_all(a.out, a.seed, render=not a.metadata_only)

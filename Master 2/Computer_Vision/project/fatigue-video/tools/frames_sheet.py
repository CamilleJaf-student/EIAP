"""
Contact sheet of video frames with their timestamps, to read event times precisely
(sync taps on the ECG electrode, start/end of exercise, gray-card position).

Usage:
    python tools/frames_sheet.py data/raw/S01_video.mov --t0 4 --t1 9 --step 0.2 --out sheet.png
    python tools/frames_sheet.py data/raw/S01_video.mov --t0 0 --t1 0 --grid   # one frame with a 10 % grid (ref_roi)
"""
import argparse

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=5.0)
    ap.add_argument("--step", type=float, default=0.2)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--width", type=int, default=320, help="thumbnail width (px)")
    ap.add_argument("--grid", action="store_true", help="draw a 10 %% grid to read ref_roi fractions")
    ap.add_argument("--rotate", type=int, default=0, choices=(0, 90, 180, 270), help="clockwise rotation (as in tests.xlsx)")
    ap.add_argument("--out", default="frames_sheet.png")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    times = np.arange(a.t0, a.t1 + 1e-9, a.step) if a.t1 > a.t0 else np.array([a.t0])
    thumbs = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, fr = cap.read()
        if not ok:
            break
        if a.rotate:
            fr = cv2.rotate(fr, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[a.rotate])
        t_real = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 - 1 / fps
        if a.grid:
            H, W = fr.shape[:2]
            for k in range(1, 10):
                cv2.line(fr, (int(W * k / 10), 0), (int(W * k / 10), H), (0, 255, 255), 1)
                cv2.line(fr, (0, int(H * k / 10)), (W, int(H * k / 10)), (0, 255, 255), 1)
                cv2.putText(fr, f"{k / 10:.1f}", (int(W * k / 10) + 3, 18), 0, 0.6, (0, 255, 255), 2)
                cv2.putText(fr, f"{k / 10:.1f}", (3, int(H * k / 10) - 4), 0, 0.6, (0, 255, 255), 2)
        w = a.width if not a.grid else fr.shape[1]
        h = int(fr.shape[0] * w / fr.shape[1])
        th = cv2.resize(fr, (w, h))
        cv2.putText(th, f"t={t_real:.3f}s", (6, h - 8), 0, 0.55 if not a.grid else 1.0, (255, 255, 255), 2)
        thumbs.append(th)
    cap.release()
    if not thumbs:
        raise SystemExit("no frame read")
    cols = min(a.cols, len(thumbs))
    rows = int(np.ceil(len(thumbs) / cols))
    h, w = thumbs[0].shape[:2]
    sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, th in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = th
    cv2.imwrite(a.out, sheet)
    print("written", a.out)


if __name__ == "__main__":
    main()

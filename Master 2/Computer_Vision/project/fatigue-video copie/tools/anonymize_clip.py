"""
Export an anonymised excerpt of a recording (face pixelated) for the oral presentation.

The face is detected with the same Viola-Jones cascade as the notebook; the box is enlarged and held when detection
fails, so that no frame shows the face. ALWAYS watch the output before sharing it.

Usage:
    python tools/anonymize_clip.py data/raw/S01_video.mov --t0 600 --t1 610 --out S01_effort_anonymised.mp4
"""
import argparse
from collections import deque

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--t0", type=float, required=True)
    ap.add_argument("--t1", type=float, required=True)
    ap.add_argument("--out", default="clip_anonymised.mp4")
    ap.add_argument("--blocks", type=int, default=8, help="pixelation blocks across the face (lower = stronger)")
    ap.add_argument("--margin", type=float, default=0.35, help="box enlargement (fraction of face width)")
    ap.add_argument("--rotate", type=int, default=0, choices=(0, 90, 180, 270), help="clockwise rotation (as printed by the notebook)")
    a = ap.parse_args()

    casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    rot = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(a.rotate)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(a.t0 * fps))
    vw, buf, last = None, deque(maxlen=9), None
    for _ in range(int((a.t1 - a.t0) * fps)):
        ok, fr = cap.read()
        if not ok:
            break
        fr = fr if rot is None else cv2.rotate(fr, rot)
        H, W = fr.shape[:2]
        s = 480 / W
        vw = vw or cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        g = cv2.cvtColor(cv2.resize(fr, (480, int(H * s))), cv2.COLOR_BGR2GRAY)
        faces = casc.detectMultiScale(g, 1.1, 4, minSize=(int(0.15 * min(g.shape)),) * 2)
        if len(faces):
            buf.append(max(faces, key=lambda f: f[2]) / s)
        if buf:
            last = np.median(np.array(buf), axis=0)
        if last is None:
            fr[:] = 0                      # no face found yet: black frame rather than an unprotected face
        else:
            x, y, w, h = last
            m = a.margin * w
            x0, y0 = int(max(0, x - m)), int(max(0, y - 1.3 * m))
            x1, y1 = int(min(W, x + w + m)), int(min(H, y + h + 1.3 * m))
            sub = fr[y0:y1, x0:x1]
            small = cv2.resize(sub, (a.blocks, max(1, int(a.blocks * sub.shape[0] / sub.shape[1]))))
            fr[y0:y1, x0:x1] = cv2.resize(small, (sub.shape[1], sub.shape[0]), interpolation=cv2.INTER_NEAREST)
        vw.write(fr)
    if vw is not None:
        vw.release()
    cap.release()
    print("written", a.out, "- check it frame by frame before sharing")


if __name__ == "__main__":
    main()

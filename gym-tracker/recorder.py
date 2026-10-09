"""
Flight recorder: a small video clip and a per-frame CSV of every session.

When a set is missed or miscounted, the clip shows what the camera saw (with
the detected skeleton drawn on) and the CSV shows what the pose model measured,
so the thresholds in reps.py can be tuned without having to reproduce it live.

Clips go to logs/clips/<start time>.mp4 and .csv: 5 frames a second, 640 px
wide. Only the newest `keep` sessions are kept.
"""

import csv
import glob
import os
import time

import cv2

from reps import FIELDS

SKELETON = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
            (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]
VIS_POINTS = {"sh": (11, 12), "el": (13, 14), "wr": (15, 16), "hip": (23, 24), "knee": (25, 26), "ank": (27, 28)}


class SessionRecorder:
    def __init__(self, log_dir, fps=5, width=640, keep=30):
        self.dir = os.path.join(log_dir, "clips")
        os.makedirs(self.dir, exist_ok=True)
        self.interval = 1.0 / fps
        self.fps = fps
        self.width = width
        self.keep = keep
        self.video = self.csv_file = self.writer = None
        self.last = None

    def start(self):
        self.stop()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.path = os.path.join(self.dir, stamp)
        self.video = None   # opened on the first frame, once the size is known
        self.csv_file = open(self.path + ".csv", "w", newline="")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(["t", "person", *FIELDS, "shoulder_y", "torso_px",
                              *[f"vis_{k}" for k in VIS_POINTS]])
        self.last = None
        self._prune()

    def frame(self, frame, f, landmarks=None):
        if self.writer is None or (self.last is not None and f.t - self.last < self.interval):
            return
        self.last = f.t
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (self.width, int(h * self.width / w)))
        if landmarks:
            sh, sw = small.shape[:2]
            pts = {i: (int(l.x * sw), int(l.y * sh)) for i, l in enumerate(landmarks) if l.visibility > 0.5}
            for a, b in SKELETON:
                if a in pts and b in pts:
                    cv2.line(small, pts[a], pts[b], (0, 255, 0), 1)
        cv2.putText(small, time.strftime("%H:%M:%S"), (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if self.video is None:
            self.video = cv2.VideoWriter(self.path + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                                         self.fps, (small.shape[1], small.shape[0]))
        self.video.write(small)

        def fmt(v):
            return "" if v is None else (round(v, 3) if isinstance(v, float) else v)
        vis = [max(landmarks[a].visibility, landmarks[b].visibility) if landmarks else ""
               for a, b in VIS_POINTS.values()]
        self.writer.writerow([round(f.t, 2), int(f.person), *[fmt(getattr(f, k)) for k in FIELDS],
                              fmt(f.shoulder_y), fmt(f.torso_px), *[fmt(v) for v in vis]])

    def stop(self):
        if self.video is not None:
            self.video.release()
        if self.csv_file is not None:
            self.csv_file.close()
        self.video = self.csv_file = self.writer = None

    def _prune(self):
        stamps = sorted({os.path.splitext(p)[0] for p in glob.glob(os.path.join(self.dir, "*.csv"))})
        for old in stamps[:-self.keep]:
            for ext in (".mp4", ".csv"):
                try:
                    os.remove(old + ext)
                except OSError:
                    pass

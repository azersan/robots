"""
Per-frame body features computed from MediaPipe Pose landmarks.

Angles come from the 3D world landmarks (metres, roughly view-independent).
Heights come from image coordinates, expressed in torso lengths so they
don't depend on how far the lifter stands from the camera. Image y points
down, so "up" values are computed as (reference_y - point_y).

Every field is None when the landmarks it needs aren't visible; the rep
counters skip frames where their signal is missing.
"""

from dataclasses import dataclass
import math

NOSE = 0
SHOULDER = (11, 12)
ELBOW = (13, 14)
WRIST = (15, 16)
HIP = (23, 24)
KNEE = (25, 26)
ANKLE = (27, 28)

MIN_VISIBILITY = 0.5


@dataclass
class Features:
    t: float                      # seconds (wall clock for live, video time for files)
    person: bool                  # shoulders + hips visible
    knee: float | None = None     # knee angle, deg (180 = straight leg); mean of visible legs
    hip: float | None = None      # shoulder-hip-knee angle, deg (180 = standing tall)
    elbow_min: float | None = None   # most-bent visible arm, deg
    elbow_mean: float | None = None  # mean of visible arms, deg
    wrist_up: float | None = None    # wrists above shoulders, torso lengths (mean)
    wrist_vs_hip: float | None = None  # wrists above hips, torso lengths (mean)
    elbow_drop: float | None = None    # elbows below shoulders, torso lengths (mean)
    # Upper-body-only signals, for when the legs are out of frame (the garage
    # camera cuts off at the shins, so knee and hip angles are often missing).
    lean: float | None = None          # torso lean from vertical, deg (0 = upright)
    shoulder_y: float | None = None    # shoulder height in the image, px (down = bigger)
    torso_px: float | None = None      # shoulder-to-hip length in the image, px
    drop: float | None = None          # filled in by reps.RepCounters: how far the
                                       # shoulders sit below standing, torso lengths


def _angle(a, b, c):
    """Angle ABC in degrees from three (x, y, z) points."""
    ba = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    bc = (c[0] - b[0], c[1] - b[1], c[2] - b[2])
    dot = sum(p * q for p, q in zip(ba, bc))
    nba = math.sqrt(sum(p * p for p in ba))
    nbc = math.sqrt(sum(p * p for p in bc))
    if nba == 0 or nbc == 0:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / (nba * nbc)))))


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def compute(landmarks, world, width, height, t):
    """Build Features from one frame's pose.

    landmarks: normalized image landmarks (x, y in 0..1, visibility)
    world: world landmarks (x, y, z in metres)
    """
    if not landmarks or not world:
        return Features(t=t, person=False)

    def vis(i):
        return landmarks[i].visibility >= MIN_VISIBILITY

    def px(i):
        return (landmarks[i].x * width, landmarks[i].y * height)

    def w(i):
        return (world[i].x, world[i].y, world[i].z)

    sides = (0, 1)
    shoulders = [s for s in sides if vis(SHOULDER[s])]
    hips = [s for s in sides if vis(HIP[s])]
    if not shoulders or not hips:
        return Features(t=t, person=False)

    sh_y = _mean(px(SHOULDER[s])[1] for s in shoulders)
    hip_y = _mean(px(HIP[s])[1] for s in hips)
    sh_x = _mean(px(SHOULDER[s])[0] for s in shoulders)
    hip_x = _mean(px(HIP[s])[0] for s in hips)
    torso = math.hypot(sh_x - hip_x, sh_y - hip_y)
    if torso < 1:
        return Features(t=t, person=False)

    legs = [s for s in sides if vis(HIP[s]) and vis(KNEE[s]) and vis(ANKLE[s])]
    knee = _mean(_angle(w(HIP[s]), w(KNEE[s]), w(ANKLE[s])) for s in legs)
    hip_sides = [s for s in sides if vis(SHOULDER[s]) and vis(HIP[s]) and vis(KNEE[s])]
    hip = _mean(_angle(w(SHOULDER[s]), w(HIP[s]), w(KNEE[s])) for s in hip_sides)

    arms = [s for s in sides if vis(SHOULDER[s]) and vis(ELBOW[s]) and vis(WRIST[s])]
    elbows = [_angle(w(SHOULDER[s]), w(ELBOW[s]), w(WRIST[s])) for s in arms]
    elbows = [e for e in elbows if e is not None]

    wrists = [s for s in sides if vis(WRIST[s])]
    wrist_y = _mean(px(WRIST[s])[1] for s in wrists)
    elbow_sides = [s for s in sides if vis(ELBOW[s])]
    elbow_y = _mean(px(ELBOW[s])[1] for s in elbow_sides)

    sh_w = [w(SHOULDER[s]) for s in shoulders]
    hip_w = [w(HIP[s]) for s in hips]
    up = tuple(_mean(p[i] for p in sh_w) - _mean(p[i] for p in hip_w) for i in range(3))
    norm = math.sqrt(sum(c * c for c in up))
    # World y points down, so upright means the hip->shoulder vector is (0, -1, 0).
    lean = math.degrees(math.acos(max(-1.0, min(1.0, -up[1] / norm)))) if norm else None

    return Features(
        t=t,
        person=True,
        lean=lean,
        shoulder_y=sh_y,
        torso_px=torso,
        knee=knee,
        hip=hip,
        elbow_min=min(elbows) if elbows else None,
        elbow_mean=_mean(elbows),
        wrist_up=(sh_y - wrist_y) / torso if wrist_y is not None else None,
        wrist_vs_hip=(hip_y - wrist_y) / torso if wrist_y is not None else None,
        elbow_drop=(elbow_y - sh_y) / torso if elbow_y is not None else None,
    )

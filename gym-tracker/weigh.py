"""
Ask Claude to read a finished set: what the lift was, and what was on the bar.

Pose tracking only knows the movement family and the rep count. The frames
saved at each rep go to Claude with that context; it names the lift in the
same style as Tony's log and counts the plates. Weights are in pounds.

Needs ANTHROPIC_API_KEY (or another credential the SDK can resolve).
"""

import base64
from typing import Literal

import anthropic
from pydantic import BaseModel

MODEL = "claude-opus-5-5"

# Names already used in the workout log, so new rows line up with old ones.
KNOWN_EXERCISES = [
    "Back Squat", "Front Squat", "Back Box Squat", "Deadlift", "Sumo Deadlift",
    "Strict Press", "Push Press", "Push Jerk", "EZ Bar Curls", "DB Hammer Curls",
]

FAMILY_HINTS = {
    "squat": "a squat pattern (knees and hips bent deeply, hands up at the shoulders or chest)",
    "hinge": "a hip hinge (deadlift-style: hips bend, arms hang straight, hands near the knees at the bottom)",
    "press": "an overhead press pattern (hands travel from the shoulders to locked out overhead)",
    "curl": "a curl pattern (elbows bend and straighten with the upper arms hanging down)",
}

SYSTEM = """You review short clips of someone lifting in their home gym, taken by a \
fixed camera. A pose tracker has already segmented one set and counted the reps; \
you get a few frames from it plus the tracker's measurements.

Your job is to fill in what the tracker can't see: the specific exercise and the \
load. The result is appended to the lifter's workout log, so be honest about \
uncertainty: if the plates can't be read (occluded, blurry, too far, side-on), say \
so in the notes, lower the confidence, and leave the total null rather than \
guessing a number.

Conventions for the log:
- Weights are pounds. A standard Olympic barbell is 45 lb unless the frames show \
otherwise (e.g. a 35 lb bar, EZ bar, trap bar). Bumper plates are often color-coded \
(red 55, blue 45, yellow 35, green 25, white 10 in lb sets; kg sets use the same \
colors for 25/20/15/10/5 kg), but read markings or sizes over colors when you can, \
and convert kg to lb.
- For dumbbells or kettlebells, the total is the weight of one implement.
- For bodyweight work, implement is "bodyweight" and the total is null.
- Name the exercise the way the existing log does when one fits."""


class PlateCount(BaseModel):
    weight_lb: float
    count_per_side: int


class SetReading(BaseModel):
    exercise: str
    implement: Literal["barbell", "dumbbell", "kettlebell", "machine", "bodyweight", "other", "unclear"]
    bar_weight_lb: float | None
    plates_per_side: list[PlateCount]
    total_weight_lb: float | None
    confidence: Literal["low", "medium", "high"]
    notes: str


def _describe(lift_set):
    lines = [
        f"Movement family from pose: {FAMILY_HINTS.get(lift_set.movement, lift_set.movement)}.",
        f"Reps counted: {lift_set.reps} (over {lift_set.t_end - lift_set.t_start:.0f} s).",
        "Per-rep measurements (angles in degrees, 180 = straight; heights in torso lengths):",
    ]
    for i, e in enumerate(lift_set.events, 1):
        s = e.stats
        parts = [f"{k}={s[k]:.2f}" if isinstance(s[k], float) else f"{k}={s[k]}"
                 for k in ("min_knee", "min_hip", "min_elbow_mean", "max_wrist_up") if k in s]
        lines.append(f"  cycle {i} ({e.t_end - e.t_start:.1f} s): " + ", ".join(parts))
    if lift_set.movement == "hinge":
        lines.append("  (A hinge set's last cycle is usually setting the bar down; it's not counted as a rep.)")
    lines.append("Exercise names already in the log: " + ", ".join(KNOWN_EXERCISES) + ".")
    return "\n".join(lines)


def _pick_frames(frames, n=3):
    if len(frames) <= n:
        return frames
    step = (len(frames) - 1) / (n - 1)
    return [frames[round(i * step)] for i in range(n)]


def read_set(lift_set, client=None):
    """Returns a SetReading, or raises anthropic.APIError on API failure."""
    client = client or anthropic.Anthropic()
    content = []
    for jpg in _pick_frames(lift_set.keyframes):
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg",
                       "data": base64.standard_b64encode(jpg).decode()},
        })
    content.append({"type": "text", "text": _describe(lift_set)})

    response = client.beta.messages.parse(
        model=MODEL,
        max_tokens=16000,
        output_config={"effort": "high"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=SYSTEM,
        messages=[{"role": "user", "content": content}],
        output_format=SetReading,
    )
    if response.stop_reason == "refusal" or response.parsed_output is None:
        raise RuntimeError(f"No reading for set {lift_set.id} (stop_reason={response.stop_reason})")
    return response.parsed_output

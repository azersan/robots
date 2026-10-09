"""
Ask Claude to read a finished set: what the lift was, and what was on the bar.

Pose tracking only knows the movement family and the rep count. The frames
saved at each rep go to Claude with that context; it names the lift in the
same style as Tony's log and counts the plates. Weights are in pounds.

Runs through the `claude` CLI (Claude Code in print mode), so it uses the
machine's Claude login rather than an API key. Frames go in as image blocks
over --input-format stream-json; the reply is constrained by --json-schema.
"""

import base64
import json
import subprocess
import tempfile
from typing import Literal

from pydantic import BaseModel

MODEL = "opus"   # latest Opus
CLAUDE_TIMEOUT = 300

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
you get a few frames from it plus the tracker's measurements. Frames from just
before and after the set usually show the bar at rest, where the plates are
sharpest; frames taken at the end of a rep show the movement but may be blurred
or have the plates partly out of the shot. The camera position is fixed.

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


# Written out by hand rather than from SetReading.model_json_schema(): the CLI
# wants a flat schema with additionalProperties: false.
SCHEMA = {
    "type": "object",
    "properties": {
        "exercise": {"type": "string"},
        "implement": {"type": "string", "enum": ["barbell", "dumbbell", "kettlebell", "machine",
                                                 "bodyweight", "other", "unclear"]},
        "bar_weight_lb": {"type": ["number", "null"]},
        "plates_per_side": {"type": "array", "items": {
            "type": "object",
            "properties": {"weight_lb": {"type": "number"}, "count_per_side": {"type": "integer"}},
            "required": ["weight_lb", "count_per_side"],
            "additionalProperties": False,
        }},
        "total_weight_lb": {"type": ["number", "null"]},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "notes": {"type": "string"},
    },
    "required": ["exercise", "implement", "bar_weight_lb", "plates_per_side",
                 "total_weight_lb", "confidence", "notes"],
    "additionalProperties": False,
}

# Keep the session to just this question: no tools, MCP servers, skills,
# settings/hooks, or saved session. Run from an empty directory so no
# CLAUDE.md gets picked up.
CLAUDE_FLAGS = [
    "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
    "--model", MODEL, "--effort", "high",
    "--tools", "", "--setting-sources", "", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
    "--disable-slash-commands", "--no-session-persistence",
]


def read_set(lift_set, claude="claude"):
    """Returns a SetReading, or raises RuntimeError if the CLI call fails."""
    content = []

    def add(label, jpg):
        content.append({"type": "text", "text": f"Frame {label}:"})
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg",
                       "data": base64.standard_b64encode(jpg).decode()},
        })

    # Frames around the set usually show the bar still (racked or on the floor),
    # which is when the plates are easiest to read; rep frames show the lift.
    for label, jpg in lift_set.context:
        add(label, jpg)
    for i, jpg in enumerate(_pick_frames(lift_set.keyframes, n=2), 1):
        add(f"at the end of a rep ({i})", jpg)
    content.append({"type": "text", "text": _describe(lift_set)})
    message = {"type": "user", "message": {"role": "user", "content": content}}

    with tempfile.TemporaryDirectory() as cwd:
        proc = subprocess.run(
            [claude, *CLAUDE_FLAGS, "--system-prompt", SYSTEM, "--json-schema", json.dumps(SCHEMA)],
            input=json.dumps(message) + "\n", capture_output=True, text=True,
            cwd=cwd, timeout=CLAUDE_TIMEOUT,
        )

    result = None
    for line in proc.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "result":
            result = event
    if result is None:
        raise RuntimeError(f"claude exited {proc.returncode} with no result: {proc.stderr.strip()[:300]}")
    if result.get("is_error") or result.get("structured_output") is None:
        raise RuntimeError(f"No reading for set {lift_set.id}: {result.get('subtype')} {str(result.get('result'))[:300]}")
    return SetReading.model_validate(result["structured_output"])

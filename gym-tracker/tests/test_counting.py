"""
Rep and set logic on synthetic pose features: each lift is a sequence of
body positions, interpolated at 15 fps and fed through the counters.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from features import Features  # noqa: E402
from reps import make_counters  # noqa: E402
from sets import SetTracker  # noqa: E402

FPS = 15

# Body positions. Angles in degrees, heights in torso lengths.
STAND = dict(knee=175, hip=175, elbow_min=170, elbow_mean=170, wrist_up=-1.0, wrist_vs_hip=0.0, elbow_drop=0.6)
# Bar on the back: hands up at the shoulders, elbows bent.
SQUAT_TOP = dict(STAND, elbow_min=70, elbow_mean=75, wrist_up=0.0, wrist_vs_hip=1.0, elbow_drop=0.3)
SQUAT_BOTTOM = dict(SQUAT_TOP, knee=85, hip=70, wrist_vs_hip=0.6)
DL_BOTTOM = dict(STAND, knee=115, hip=80, wrist_vs_hip=-0.9)
PRESS_RACK = dict(STAND, elbow_min=55, elbow_mean=60, wrist_up=0.05, wrist_vs_hip=1.0, elbow_drop=0.3)
PRESS_TOP = dict(PRESS_RACK, elbow_min=170, elbow_mean=172, wrist_up=1.0, wrist_vs_hip=2.0, elbow_drop=-0.3)
CURL_TOP = dict(STAND, elbow_min=45, elbow_mean=50, wrist_up=-0.1, wrist_vs_hip=0.9)


def timeline(*steps):
    """steps: (pose, seconds to move there). Yields Features at FPS."""
    t, frames = 0.0, []
    prev = steps[0][0]
    for pose, secs in steps:
        n = max(1, int(secs * FPS))
        for i in range(1, n + 1):
            a = i / n
            vals = {k: prev[k] + (pose[k] - prev[k]) * a for k in pose}
            t += 1 / FPS
            frames.append(Features(t=t, person=True, **vals))
        prev = pose
    return frames


def reps(top, bottom, n, down=1.2, up=1.2, pause=0.5):
    steps = []
    for _ in range(n):
        steps += [(bottom, down), (top, up), (top, pause)]
    return steps


def run(frames, tracker=None):
    counters = make_counters()
    tracker = tracker or SetTracker()
    events, sets = [], []
    for f in frames:
        for c in counters:
            ev = c.update(f)
            if ev:
                events.append(ev)
                done = tracker.on_rep(ev)
                if done:
                    sets.append(done)
        done = tracker.tick(f.t)
        if done:
            sets.append(done)
    done = tracker.flush()
    if done:
        sets.append(done)
    return events, sets


def test_back_squat_five_reps():
    events, sets = run(timeline((STAND, 1), (SQUAT_TOP, 1), *reps(SQUAT_TOP, SQUAT_BOTTOM, 5), (STAND, 1)))
    assert [e.movement for e in events] == ["squat"] * 5
    assert len(sets) == 1 and sets[0].movement == "squat" and sets[0].reps == 5


def test_deadlift_drops_the_put_down_cycle():
    # Walk up, bend to grip, 3 pulls, set it down, stand up empty-handed.
    frames = timeline((STAND, 1), (DL_BOTTOM, 1.5), (STAND, 1.5),
                      (DL_BOTTOM, 1.5), (STAND, 1.5), (DL_BOTTOM, 1.5), (STAND, 1.5),
                      (DL_BOTTOM, 1.5), (STAND, 1.5))
    events, sets = run(frames)
    assert [e.movement for e in events] == ["hinge"] * 4
    assert len(sets) == 1 and sets[0].reps == 3


def test_strict_press_counts_and_is_not_a_curl():
    frames = timeline((STAND, 1), (PRESS_RACK, 1), *reps(PRESS_RACK, PRESS_TOP, 4, down=1.0, up=1.0), (STAND, 1))
    events, sets = run(frames)
    assert {e.movement for e in events} == {"press"}
    assert len(sets) == 1 and sets[0].reps == 4


def test_curls():
    events, sets = run(timeline((STAND, 1), *reps(STAND, CURL_TOP, 8, down=0.8, up=0.8, pause=0.2)))
    assert [e.movement for e in events] == ["curl"] * 8
    assert sets[0].reps == 8


def test_squat_is_not_a_hinge_and_hinge_is_not_a_squat():
    sq, _ = run(timeline((SQUAT_TOP, 1), *reps(SQUAT_TOP, SQUAT_BOTTOM, 3)))
    dl, _ = run(timeline((STAND, 1), *reps(STAND, DL_BOTTOM, 3)))
    assert {e.movement for e in sq} == {"squat"}
    assert {e.movement for e in dl} == {"hinge"}


def test_partial_rep_does_not_count():
    half = dict(SQUAT_BOTTOM, knee=130, hip=120)
    events, _ = run(timeline((SQUAT_TOP, 1), (half, 1), (SQUAT_TOP, 1)))
    assert events == []


def test_rest_splits_sets_and_single_reps_are_dropped():
    frames = timeline((SQUAT_TOP, 1), *reps(SQUAT_TOP, SQUAT_BOTTOM, 3),
                      (SQUAT_TOP, 40),                       # rest longer than rest_gap
                      *reps(SQUAT_TOP, SQUAT_BOTTOM, 3),
                      (SQUAT_TOP, 40),
                      *reps(SQUAT_TOP, SQUAT_BOTTOM, 1))      # a lone rep: noise
    _, sets = run(frames)
    assert [s.reps for s in sets] == [3, 3]


def test_signal_dropout_resets_counter():
    frames = timeline((SQUAT_TOP, 1), (SQUAT_BOTTOM, 1))
    t = frames[-1].t
    gap = [Features(t=t + 0.1 * i, person=False) for i in range(1, 50)]   # 5 s of nothing
    back = [Features(t=gap[-1].t + f.t, person=True,
                     **{k: getattr(f, k) for k in SQUAT_TOP}) for f in timeline((SQUAT_TOP, 1))]
    events, _ = run(frames + gap + back)
    assert events == []

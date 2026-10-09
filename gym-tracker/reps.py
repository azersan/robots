"""
Rep counting from per-frame Features.

Each movement family has a CycleCounter watching one signal (a joint angle or
a height). A rep is: leave the rest zone -> pass the turn threshold -> come
back to rest. Hysteresis between the two thresholds keeps jitter from
double-counting. When a cycle completes, the movement's gates check the
rep's stats (e.g. "wrists stayed above the hips at the bottom") so a squat
doesn't also count as a deadlift.

Pose only identifies the *family* (squat / hinge / press / curl). The exact
lift name (Back vs Front Squat, Strict vs Push Press) is decided later by
Claude looking at the frames, with these stats as context.
"""

from dataclasses import dataclass, field

FIELDS = ("knee", "hip", "elbow_min", "elbow_mean", "wrist_up", "wrist_vs_hip", "elbow_drop")

# Lose the signal for this long and the counter forgets where it was.
SIGNAL_TIMEOUT = 3.0


@dataclass
class RepEvent:
    movement: str
    t_start: float
    t_end: float
    stats: dict = field(default_factory=dict)  # min_/max_<field>, turn_<field>


class CycleCounter:
    def __init__(self, movement, signal, rest, turn, gates=(), min_duration=0.4, max_duration=15.0):
        self.movement = movement
        self.signal = signal
        self.rest = rest
        self.turn = turn
        # True when the signal falls into the rep (knee angle closing),
        # False when it rises (wrists going overhead).
        self.falls = turn < rest
        self.gates = gates
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.reset()

    def reset(self):
        self.state = "unknown"   # unknown -> rest -> moving -> turned -> (rep) rest
        self.last_seen = None
        self._start_rep(None)

    def _start_rep(self, t):
        self.t_start = t
        self.mins, self.maxs, self.turn_snapshot, self.extreme = {}, {}, {}, None

    def _at_rest(self, v):
        return v >= self.rest if self.falls else v <= self.rest

    def _past_turn(self, v):
        return v <= self.turn if self.falls else v >= self.turn

    def _more_extreme(self, v):
        if self.extreme is None:
            return True
        return v < self.extreme if self.falls else v > self.extreme

    def _accumulate(self, f, v):
        for name in FIELDS:
            x = getattr(f, name)
            if x is None:
                continue
            self.mins[name] = min(self.mins.get(name, x), x)
            self.maxs[name] = max(self.maxs.get(name, x), x)
        if self._more_extreme(v):
            self.extreme = v
            self.turn_snapshot = {n: getattr(f, n) for n in FIELDS}

    def update(self, f):
        """Feed one frame; returns a RepEvent when a valid rep completes."""
        v = getattr(f, self.signal)
        if v is None:
            if self.last_seen is not None and f.t - self.last_seen > SIGNAL_TIMEOUT:
                self.reset()
            return None
        self.last_seen = f.t

        if self.state == "unknown":
            if self._at_rest(v):
                self.state = "rest"
            return None

        if self.state == "rest":
            if not self._at_rest(v):
                self.state = "moving"
                self._start_rep(f.t)
                self._accumulate(f, v)
            return None

        self._accumulate(f, v)
        if self.state == "moving" and self._past_turn(v):
            self.state = "turned"
        if self._at_rest(v):
            completed = self.state == "turned"
            self.state = "rest"
            if completed:
                return self._finish(f.t)
        elif f.t - self.t_start > self.max_duration:
            # Parked mid-range too long (resting in the hole, walking off): drop it.
            self.state = "unknown"
        return None

    def _finish(self, t_end):
        duration = t_end - self.t_start
        if not (self.min_duration <= duration <= self.max_duration):
            return None
        stats = {f"min_{k}": v for k, v in self.mins.items()}
        stats.update({f"max_{k}": v for k, v in self.maxs.items()})
        stats.update({f"turn_{k}": v for k, v in self.turn_snapshot.items() if v is not None})
        if not all(gate(stats) for gate in self.gates):
            return None
        return RepEvent(self.movement, self.t_start, t_end, stats)


def _lt(key, limit):
    return lambda s: s.get(key) is not None and s[key] < limit


def _gt(key, limit):
    return lambda s: s.get(key) is not None and s[key] > limit


def make_counters():
    """The movement families we recognise, with thresholds and gates."""
    return [
        # Squat: knees close well past 90-ish and reopen, with the hands up
        # at the shoulders/chest (bar on back or front rack, goblet).
        CycleCounter("squat", "knee", rest=155, turn=110, gates=(
            _gt("turn_wrist_vs_hip", 0.25),   # hands above hips in the hole
            _lt("turn_hip", 130),             # hips actually flexed too
        )),
        # Hinge (deadlift, trap-bar deadlift, RDL): hips close and reopen with
        # the arms hanging and the hands down near the knees. Knee bend isn't
        # gated - a trap-bar pull bends the knees as much as a squat; the hand
        # position is what separates the two.
        CycleCounter("hinge", "hip", rest=155, turn=115, gates=(
            _lt("turn_wrist_vs_hip", -0.3),   # hands well below hips at the bottom
            _gt("min_elbow_mean", 120),       # arms stay (roughly) straight
        )),
        # Overhead press: wrists go from shoulder height to locked out overhead.
        CycleCounter("press", "wrist_up", rest=0.35, turn=0.65, gates=(
            _gt("turn_elbow_mean", 145),      # elbows locked out at the top
        )),
        # Curl: elbow closes and reopens with the upper arm hanging down and
        # the hands never going overhead (rules out the press lowering phase).
        CycleCounter("curl", "elbow_min", rest=140, turn=75, gates=(
            _lt("max_wrist_up", 0.35),
            _gt("turn_elbow_drop", 0.35),
        )),
    ]


def reps_in_set(movement, cycles):
    """Turn completed cycles into reps for a finished set.

    A hinge set has one cycle more than it has reps: walking up and bending
    to grip the bar, then pulling, is rep 1, but setting the bar down and
    standing up empty-handed afterwards also looks like a full cycle.
    """
    if movement == "hinge" and cycles >= 2:
        return cycles - 1
    return cycles

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

FIELDS = ("knee", "hip", "elbow_min", "elbow_mean", "wrist_up", "wrist_vs_hip", "elbow_drop",
          "lean", "drop")

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


class StandingReference:
    """Where the shoulders sit when standing, so a squat can be seen as a drop.

    Keeps the last few seconds of shoulder positions; the highest one is
    "standing". drop = how far below that the shoulders are now, in torso
    lengths. Works from the upper body alone.
    """

    def __init__(self, window=10.0):
        self.window = window
        self.samples = []   # (t, shoulder_y, torso_px)

    def update(self, f):
        if f.shoulder_y is None or not f.torso_px:
            return None
        self.samples = [s for s in self.samples if f.t - s[0] <= self.window]
        self.samples.append((f.t, f.shoulder_y, f.torso_px))
        _, ref_y, ref_torso = min(self.samples, key=lambda s: s[1])
        return (f.shoulder_y - ref_y) / ref_torso


class RepCounters:
    """All the movement counters, fed one frame of Features at a time."""

    def __init__(self):
        self.reference = StandingReference()
        self.counters = make_counters()

    def update(self, f):
        f.drop = self.reference.update(f)
        events = []
        for counter in self.counters:
            event = counter.update(f)
            if event is not None:
                events.append(event)
        return events


def make_counters():
    """The movement families we recognise, with thresholds and gates.

    Squat and hinge use upper-body signals (shoulder drop, torso lean): the
    garage camera cuts off at the shins, so knee and hip angles, which need
    the ankles and knees, are usually missing there.
    """
    return [
        # Squat: the shoulders drop well below standing and come back up, with
        # the hands no lower than about hip height: up at the shoulders with a
        # bar, around the hips in a bodyweight squat (measured -0.1 to -0.2 in
        # the garage).
        CycleCounter("squat", "drop", rest=0.2, turn=0.5, gates=(
            _gt("turn_wrist_vs_hip", -0.4),
        )),
        # Hinge (deadlift, trap-bar deadlift, RDL): the torso tips forward and
        # comes back up with the arms hanging and the hands down near the knees.
        # The hand position is what separates it from a squat, which also leans.
        CycleCounter("hinge", "lean", rest=25, turn=45, gates=(
            _lt("turn_wrist_vs_hip", -0.45),  # hands down near the knees at the bottom
            _gt("min_elbow_mean", 120),       # arms stay (roughly) straight
        )),
        # Overhead press: wrists go from shoulder height to locked out overhead.
        CycleCounter("press", "wrist_up", rest=0.35, turn=0.65, gates=(
            # Elbows (nearly) locked out at the top. 120 not 160+: seen from an
            # angle, a locked-out arm measures 126-146 (garage camera, 2026-10-09);
            # hands 0.65 torso lengths above the shoulders already means overhead.
            _gt("turn_elbow_mean", 120),
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

"""
Group rep events into sets.

A set is consecutive reps of one movement. It ends when the lifter rests
longer than `rest_gap`, leaves the frame, or the session ends. Sets shorter
than `min_reps` are discarded as noise (bending to load a plate looks a lot
like one deadlift rep).
"""

from dataclasses import dataclass, field
import itertools

from reps import reps_in_set

_ids = itertools.count(1)


@dataclass
class LiftSet:
    movement: str
    events: list = field(default_factory=list)
    keyframes: list = field(default_factory=list)   # JPEG bytes, one per rep
    id: int = field(default_factory=lambda: next(_ids))

    @property
    def t_start(self):
        return self.events[0].t_start

    @property
    def t_end(self):
        return self.events[-1].t_end

    @property
    def reps(self):
        return reps_in_set(self.movement, len(self.events))


class SetTracker:
    def __init__(self, rest_gap=25.0, min_reps=2, hot_window=8.0, max_keyframes=6):
        self.rest_gap = rest_gap
        self.min_reps = min_reps
        # While a set is "hot" (last rep this recent), a rep of a different
        # movement is treated as a misread rather than starting a new set.
        self.hot_window = hot_window
        self.max_keyframes = max_keyframes
        self.current = None

    def on_rep(self, event, keyframe=None):
        """Record a rep. Returns a finished LiftSet if this rep closed the previous one."""
        finished = None
        cur = self.current
        if cur is not None:
            gap = event.t_start - cur.t_end
            if event.movement != cur.movement and gap < self.hot_window:
                return None
            if event.movement != cur.movement or gap > self.rest_gap:
                finished = self._close()
        if self.current is None:
            self.current = LiftSet(event.movement)
        self.current.events.append(event)
        if keyframe is not None and len(self.current.keyframes) < self.max_keyframes:
            self.current.keyframes.append(keyframe)
        return finished

    def tick(self, now):
        """Close the current set if the rest gap has passed. Returns it if kept."""
        if self.current is not None and now - self.current.t_end > self.rest_gap:
            return self._close()
        return None

    def flush(self):
        """End of session: close whatever is open."""
        return self._close() if self.current is not None else None

    def _close(self):
        s, self.current = self.current, None
        return s if s.reps >= self.min_reps else None

"""
Show tracker updates on the garage LED clock (Ulanzi TC002 running AWTRIX NG).

Each update is an AWTRIX notification named "gym" with stack=false, so a new
one replaces the last instead of queueing behind it. Text wider than the
52-pixel panel scrolls, so each message has shorter fallbacks and the first
one that fits is sent (widths measured on the clock with tools/clock_screen.py).

Posts are fire-and-forget on a background thread with a short timeout, so a
slow or unplugged clock never stalls the video loop.
"""

import concurrent.futures
import json
import urllib.request

DEFAULT_HOST = "192.168.4.37"

WHITE, GREEN, YELLOW = "#FFFFFF", "#00FF00", "#FFD000"

PANEL_WIDTH = 52
# Glyph widths of AWTRIX's double-size font on the TC002, plus a 2 px gap
# between characters. Most glyphs are 6 px.
GLYPH_WIDTH = {" ": 2, ".": 2, "Q": 8, "M": 10, "W": 10}

NAMES = {"squat": ("SQUAT", "SQ"), "hinge": ("DL", "DL"), "press": ("PRESS", "PR"), "curl": ("CURL", "CU")}


def text_width(text):
    return sum(GLYPH_WIDTH.get(c, 6) for c in text) + 2 * (len(text) - 1)


def fit(*candidates):
    """The first candidate that fits the panel without scrolling (else the last)."""
    for text in candidates:
        if text_width(text) <= PANEL_WIDTH:
            return text
    return candidates[-1]


class Display:
    def __init__(self, host=DEFAULT_HOST):
        self.url = f"http://{host}/api/v1/notifications"
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def show(self, text, color=WHITE, seconds=30):
        body = json.dumps({
            "name": "gym",
            "text": text.upper(),
            "textColor": color,
            "durationMs": int(seconds * 1000),
            "stack": False,
        }).encode()
        self._pool.submit(self._post, body)

    def _post(self, body):
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=2).read()
        except OSError:
            pass  # clock off or unreachable: the tracker carries on

    def clear(self):
        """Take the gym notification off the clock."""
        req = urllib.request.Request(self.url + "/gym", method="DELETE")
        self._pool.submit(lambda: self._send(req))

    def _send(self, req):
        try:
            urllib.request.urlopen(req, timeout=2).read()
        except OSError:
            pass

    # -- tracker events --------------------------------------------------------

    def rep(self, movement, count):
        name, abbr = NAMES.get(movement, (movement.upper(), movement.upper()[:2]))
        self.show(fit(f"{name} {count}", f"{abbr} {count}"))                # SQUAT 3, SQ 10

    def set_done(self, reps):
        self.show(fit(f"{reps} REPS", f"{reps}REPS"), seconds=60)            # 5 REPS

    def set_read(self, reps, weight_lb, confidence):
        if weight_lb is None:
            self.show(fit(f"{reps} X ?", f"{reps}X?"), YELLOW, seconds=90)   # weight unread
            return
        color = GREEN if confidence == "high" else YELLOW
        w = f"{weight_lb:g}"
        self.show(fit(f"{reps} X {w}", f"{reps}X{w}", f"{reps}X{round(weight_lb)}"),
                  color, seconds=90)                                         # 5 X 185 (lb)

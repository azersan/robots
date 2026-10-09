"""
Show tracker updates on the garage LED clock (Ulanzi, custom-app API).

Posts are fire-and-forget on a background thread with a short timeout, so a
slow or unplugged display never stalls the video loop.
"""

import concurrent.futures
import json
import urllib.request

DEFAULT_URL = "http://192.168.4.37/api/custom?name=garage"

WHITE, GREEN, YELLOW = "#FFFFFF", "#00FF00", "#FFD000"

# The stock TC002 firmware fits about 8 characters at fontHeight 10 and its
# font has no "?", so every message stays <= 8 characters of plain A-Z/0-9.
# Weights are always pounds, so "LB" is left off.
SHORT_NAMES = {"squat": "SQUAT", "hinge": "DL", "press": "PRESS", "curl": "CURL"}


class Display:
    def __init__(self, url=DEFAULT_URL):
        self.url = url
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def show(self, text, color=WHITE, seconds=30):
        body = json.dumps({
            "duration": seconds,
            "text": [{"content": text.upper(), "fontHeight": 10, "y": 3, "x": -1000,
                      "align": "center", "rect": [0, 0, 52, 16], "color": color}],
        }).encode()
        self._pool.submit(self._post, body)

    def _post(self, body):
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=2).read()
        except OSError:
            pass  # display off or unreachable: the tracker carries on

    # -- tracker events --------------------------------------------------------

    def rep(self, movement, count):
        self.show(f"{SHORT_NAMES.get(movement, movement)} {count}")        # SQUAT 3, DL 5

    def set_done(self, reps):
        self.show(f"{reps} REPS", seconds=60)                                # 5 REPS

    def set_read(self, reps, weight_lb, confidence):
        if weight_lb is None:
            self.show(f"{reps} REPS", YELLOW, seconds=90)                    # weight unread
        else:
            color = GREEN if confidence == "high" else YELLOW
            self.show(f"{reps} X {weight_lb:g}", color, seconds=90)          # 5 X 185

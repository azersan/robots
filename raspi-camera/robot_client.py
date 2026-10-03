"""
Mac-side client for the Pi robot control server (robot_server.py).

Sends drive commands over HTTP without ever stalling the caller (e.g. a CV
loop). A single background worker thread owns the network: callers just
overwrite a "latest pending command" slot, and the worker sends whatever is
pending, at most MAX_RATE_HZ times per second. This design guarantees:

  - The LAST command in a burst is always sent (coalesced, not dropped),
    so a final "straighten out" or "stop" can't be rate-limited away.
  - Commands arrive in order (one thread, one connection), so a stale
    drive command can never overtake a stop.

Best-effort on failures: if a request fails, the Pi's watchdog stops the
motors after COMMAND_TIMEOUT, so dropping commands is safe.

Usage:
    from robot_client import RobotClient
    bot = RobotClient()              # defaults to pibot5-2g.local:8080
    bot.drive(steer=0.3, speed=0.5)  # curve right at half speed
    bot.tank(left=0.8, right=-0.8)   # spin right in place
    bot.stop()
"""

import threading
import time
import requests

DEFAULT_HOST = "pibot5-2g.local"
DEFAULT_PORT = 8080
MAX_RATE_HZ = 15  # coalesce command spam to this rate


class RobotClient:
    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT, timeout=0.3):
        self.base = f"http://{host}:{port}"
        self.timeout = timeout
        self._session = requests.Session()  # connection reuse
        self._min_interval = 1.0 / MAX_RATE_HZ
        self._cond = threading.Condition()
        self._pending = None  # (path, payload) - newest unsent command
        self._worker = threading.Thread(target=self._send_loop, daemon=True)
        self._worker.start()

    # -- internal --------------------------------------------------------
    def _send_loop(self):
        last_send = 0.0
        while True:
            with self._cond:
                while self._pending is None:
                    self._cond.wait()
            # Rate-limit BEFORE popping: commands arriving during the pause
            # overwrite _pending, so we always send the newest one.
            delay = self._min_interval - (time.monotonic() - last_send)
            if delay > 0:
                time.sleep(delay)
            with self._cond:
                path, payload = self._pending  # only this thread clears it
                self._pending = None
            try:
                self._session.post(self.base + path, json=payload,
                                   timeout=self.timeout)
            except requests.RequestException:
                pass  # best-effort; Pi watchdog halts motors if we go silent
            last_send = time.monotonic()

    def _submit(self, path, payload):
        with self._cond:
            self._pending = (path, payload)
            self._cond.notify()

    # -- public ----------------------------------------------------------
    def drive(self, steer=0.0, speed=0.0):
        """Curve while driving. steer -1..1 (left..right), speed -1..1
        (reverse..forward)."""
        self._submit("/drive", {"steer": float(steer), "speed": float(speed)})

    def tank(self, left=0.0, right=0.0):
        """Direct per-side control, each -1..1 (reverse..forward)."""
        self._submit("/tank", {"left": float(left), "right": float(right)})

    def stop(self):
        """Stop the motors. Replaces any pending command."""
        self._submit("/stop", {})

    def health(self):
        """Synchronous health check. Returns dict or None on failure."""
        try:
            r = self._session.get(self.base + "/health", timeout=self.timeout)
            return r.json()
        except requests.RequestException:
            return None

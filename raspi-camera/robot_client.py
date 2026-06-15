"""
Mac-side client for the Pi robot control server (robot_server.py).

Sends drive commands over HTTP, non-blocking and rate-limited so it never
stalls the caller (e.g. a CV loop). Best-effort: if a request fails, the Pi's
watchdog stops the motors after COMMAND_TIMEOUT, so dropping commands is safe.

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
        self._lock = threading.Lock()
        self._last_send = 0.0
        self._min_interval = 1.0 / MAX_RATE_HZ

    # -- internal --------------------------------------------------------
    def _do_post(self, path, payload):
        try:
            requests.post(self.base + path, json=payload, timeout=self.timeout)
        except requests.RequestException:
            pass  # best-effort; Pi watchdog halts motors if we go silent

    def _post_async(self, path, payload):
        threading.Thread(target=self._do_post, args=(path, payload), daemon=True).start()

    def _throttle(self):
        now = time.time()
        with self._lock:
            if now - self._last_send < self._min_interval:
                return False
            self._last_send = now
        return True

    # -- public ----------------------------------------------------------
    def drive(self, steer=0.0, speed=0.0):
        """Curve while driving. steer -1..1 (left..right), speed 0..1."""
        if not self._throttle():
            return
        self._post_async("/drive", {"steer": float(steer), "speed": float(speed)})

    def tank(self, left=0.0, right=0.0):
        """Direct per-side control, each -1..1 (reverse..forward)."""
        if not self._throttle():
            return
        self._post_async("/tank", {"left": float(left), "right": float(right)})

    def stop(self):
        """Stop the motors. Always sent (bypasses throttle)."""
        self._post_async("/stop", {})

    def health(self):
        """Synchronous health check. Returns dict or None on failure."""
        try:
            r = requests.get(self.base + "/health", timeout=self.timeout)
            return r.json()
        except requests.RequestException:
            return None

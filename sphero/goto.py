"""Steer the BOLT+ through image-space waypoints using the eyes_server tracker.

  python goto.py 845,390 845,560        # drive to each (x, y) pixel in turn

Continuous closed loop: keep the robot rolling slowly and re-aim it ~4x/s
from the latest tracked position. The mapping from robot heading to image
direction ("offset": image angle = heading + offset) isn't known up front and
varies with perspective, so it's re-estimated from how the robot actually
moves. Short start/stop bursts don't work - the BOLT+ needs time to spin up.
"""
import argparse
import json
import math
import time
import urllib.request
from collections import deque

BASE = "http://127.0.0.1:8770"


def get_state():
    with urllib.request.urlopen(BASE + "/state", timeout=2) as r:
        return json.load(r)


def post(path):
    urllib.request.urlopen(urllib.request.Request(BASE + path, method="POST"), timeout=2).read()


def wrap(a):
    return (a + 180) % 360 - 180


def fix():
    """(x, y, t_fix) of the latest tracker fix, or None if not locked."""
    s = get_state()
    b = s.get("bolt")
    if s.get("tracking") != "locked" or not b:
        return None
    return b["x"], b["y"], s["t"] - b["age_s"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("waypoints", nargs="+", help="x,y in full-frame pixels")
    p.add_argument("--offset", type=float, default=40.0,
                   help="initial guess: image angle = heading + offset (deg)")
    p.add_argument("--tol", type=float, default=15.0, help="arrival radius, px")
    # Sustained, the BOLT+ is fast: speed 75 crosses the frame in ~2s and
    # outruns the blink tracker. Keep it slow.
    p.add_argument("--vmin", type=int, default=25, help="slowest speed that still rolls")
    p.add_argument("--vmax", type=int, default=40)
    p.add_argument("--timeout", type=float, default=60.0, help="per waypoint, s")
    args = p.parse_args()
    wps = [tuple(map(float, w.split(","))) for w in args.waypoints]
    offset = args.offset
    hist = deque(maxlen=20)   # (t_fix, x, y, heading commanded at that time)
    heading = None

    try:
        for wi, (tx, ty) in enumerate(wps):
            t_start, t_good = time.time(), time.time()
            while True:
                if time.time() - t_start > args.timeout:
                    print(f"timed out on waypoint {wi}")
                    return
                f = fix()
                now = time.time()
                if f is None or now - f[2] > 0.8:
                    post("/bolt/stop")  # never drive blind
                    if now - t_good > 5:
                        print("lost track; stopping")
                        return
                    time.sleep(0.1)
                    continue
                t_good = now
                x, y, tf = f
                if not hist or tf > hist[-1][0]:
                    hist.append((tf, x, y, heading))
                dist = math.hypot(tx - x, ty - y)
                if dist < args.tol:
                    post("/bolt/stop")
                    print(f"reached waypoint {wi} ({tx:.0f},{ty:.0f}) at ({x},{y})  offset {offset:.0f}")
                    time.sleep(0.8)  # let it coast to a stop before the next leg
                    break
                # Learn the heading->image mapping from the last ~0.7s of motion,
                # if we were commanding one heading the whole time.
                old = [h for h in hist if tf - 0.9 < h[0] <= tf - 0.5]
                if old and heading is not None:
                    t0, x0, y0, h0 = old[0]
                    moved = math.hypot(x - x0, y - y0)
                    if moved > 10 and h0 is not None and abs(wrap(h0 - heading)) < 15:
                        seen = math.degrees(math.atan2(y - y0, x - x0))
                        offset = (offset + 0.3 * wrap(seen - heading - offset)) % 360
                want = math.degrees(math.atan2(ty - y, tx - x))
                heading = round((want - offset) % 360)
                speed = round(min(args.vmax, max(args.vmin, args.vmin + 0.1 * (dist - 30))))
                post(f"/bolt/drive?heading={heading}&speed={speed}&dur=0.5")
                print(f"  wp{wi} at ({x},{y}) dist {dist:4.0f}  heading {heading:3d}  "
                      f"speed {speed}  offset {offset:4.0f}")
                time.sleep(0.25)
        print("done")
    finally:
        post("/bolt/stop")


if __name__ == "__main__":
    main()

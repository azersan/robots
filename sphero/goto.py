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


# The camera sees the floor at a slant, so depth (image y) is foreshortened
# relative to sideways motion. Angles are computed with y stretched by this
# factor, which makes heading->image offset roughly direction-independent.
Y_STRETCH = 2.0


def floor_angle(dx, dy):
    return math.degrees(math.atan2(dy * Y_STRETCH, dx)) % 360


def wrap(a):
    return (a + 180) % 360 - 180


def fix():
    """(x, y, t_fix) of the latest tracker fix, or None if not locked."""
    s = get_state()
    b = s.get("bolt")
    if s.get("tracking") != "locked" or not b:
        return None
    return b["x"], b["y"], s["t"] - b["age_s"]


def calibrate(speed=30, secs=1.0):
    """Roll briefly at heading 0 and return the observed offset (deg), or None.

    The heading frame resets on every reconnect, so measure it each run."""
    f0 = fix()
    if f0 is None:
        return None
    post("/bolt/heading?deg=0")
    time.sleep(0.8)
    t0 = time.time()
    while time.time() - t0 < secs:
        post(f"/bolt/drive?heading=0&speed={speed}&dur=0.4")
        time.sleep(0.2)
    post("/bolt/stop")
    time.sleep(1.0)  # coast + let the tracker catch up
    f1 = fix()
    if f1 is None:
        return None
    dx, dy = f1[0] - f0[0], f1[1] - f0[1]
    moved = math.hypot(dx, dy)
    print(f"calibration: moved {moved:.0f}px from {f0[:2]} to {f1[:2]}")
    return floor_angle(dx, dy) if moved > 8 else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("waypoints", nargs="+", help="x,y in full-frame pixels")
    p.add_argument("--offset", type=float, default=None,
                   help="image angle = heading + offset (deg); measured if omitted")
    p.add_argument("--tol", type=float, default=15.0, help="arrival radius, px")
    # Sustained, the BOLT+ is fast: speed 75 crosses the frame in ~2s and
    # outruns the blink tracker. Keep it slow.
    p.add_argument("--vmin", type=int, default=25, help="slowest speed that still rolls")
    p.add_argument("--vmax", type=int, default=40)
    p.add_argument("--timeout", type=float, default=60.0, help="per waypoint, s")
    args = p.parse_args()
    wps = [tuple(map(float, w.split(","))) for w in args.waypoints]
    offset = args.offset
    for secs in (1.0, 1.6):
        if offset is not None:
            break
        offset = calibrate(secs=secs)
    if offset is None:
        raise SystemExit("couldn't measure heading offset (robot not moving / not tracked)")
    print(f"offset {offset:.0f}")
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
                old = [h for h in hist if tf - 1.0 < h[0] <= tf - 0.6]
                if old and heading is not None:
                    t0, x0, y0, h0 = old[0]
                    moved = math.hypot(x - x0, y - y0)
                    steady = all(h[3] is not None and abs(wrap(h[3] - heading)) < 12
                                 for h in hist if h[0] >= t0)
                    if moved > 12 and steady:
                        seen = floor_angle(x - x0, y - y0)
                        offset = (offset + 0.2 * wrap(seen - heading - offset)) % 360
                want = floor_angle(tx - x, ty - y)
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

"""Quick end-to-end check without the gateway: build, drive, latch, render.

    python tools/smoke.py [--scene flat|anna_pl|anna_pl_scan] [--out /tmp/simpg]
"""

import argparse
import math
import os
import sys
import time

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simpg.sim import Sim  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="flat")
    ap.add_argument("--out", default="/tmp/simpg")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    sim = Sim(args.scene)
    sim.cmd_timeout = 0  # commands below are held across many frames
    print(f"built {args.scene} in {time.time() - t0:.1f}s")

    def snap(tag):
        for cam in ("front", "chase", "overhead"):
            rgb, _ = sim.render(cam)
            Image.fromarray(rgb).save(f"{args.out}/{tag}_{cam}.png")

    snap("start")
    s = sim.get_state()["car"]
    print(f"start  x={s['x']:.2f} y={s['y']:.2f} z={s['z']:.3f} yaw={math.degrees(s['yaw']):.0f}")

    t0 = time.time()
    sim.set_drive(1.0, 0.0)
    sim.step(120)
    s = sim.get_state()["car"]
    wall = time.time() - t0
    print(f"after 2s @1m/s: x={s['x']:.2f} y={s['y']:.2f} v={s['v']:.2f} upright={s['upright']:.3f} "
          f"({120 / wall:.0f} sim fps)")

    sim.set_drive(0.0, 1.0)
    sim.step(60)
    s = sim.get_state()["car"]
    print(f"after 1s turn @1rad/s: yaw={math.degrees(s['yaw']):.0f}deg w={s['w']:.2f}")

    sim.set_drive(0.0, 0.0)
    sim.step(30)
    snap("end")

    sim.reset(randomize=False)
    s = sim.get_state()["car"]
    print(f"reset  x={s['x']:.2f} y={s['y']:.2f}")
    print(f"nearest latch: {sim.nearest_latch()}")
    print(f"images in {args.out}")


if __name__ == "__main__":
    main()

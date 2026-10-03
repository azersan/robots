"""Render what the robot's front camera sees while lined up behind a bin, at several distances.

    python tools/robot_views.py [--scene anna_pl] [--out docs/robot_view]

Used for the reference images in ROBOT.md, and to check when the latch bar drops out of view.
Distances are from the hook to the latch bar, with the robot square to the bin and centered.
"""

import argparse
import math
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simpg.sim import HOOK_LOCAL, Sim  # noqa: E402

DISTANCES = [3.0, 1.0, 0.5, 0.3, 0.2, 0.15, 0.1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="anna_pl")
    ap.add_argument("--bin", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "docs", "robot_view"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out + "_x")), exist_ok=True)

    sim = Sim(args.scene)
    b = sim.get_state()["bins"][args.bin]
    lx, ly, _ = b["latch"]
    n = np.array([math.cos(b["yaw"]), math.sin(b["yaw"])])
    yaw = b["yaw"] + math.pi  # facing the bin
    for d in DISTANCES:
        cx, cy = np.array([lx, ly]) + n * (HOOK_LOCAL[0] + d)
        sim.place_car(cx, cy, yaw)
        rgb, _ = sim.render("front")
        name = f"{args.out}_{int(round(d * 100)):03d}cm.jpg"
        Image.fromarray(rgb).save(name, quality=88)
        print(f"{d:5.2f} m -> {os.path.basename(name)}")


if __name__ == "__main__":
    main()

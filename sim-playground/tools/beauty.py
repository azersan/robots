"""Render a few showcase frames of a scene from free cameras.

    python tools/beauty.py [--scene anna_pl] [--out /tmp/simpg_beauty]
"""

import argparse
import math
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simpg.sim import Sim  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="anna_pl")
    ap.add_argument("--out", default="/tmp/simpg_beauty")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sim = Sim(args.scene)
    st = sim.get_state()
    car = st["car"]
    b0, b1 = st["bins"][0], st["bins"][1]
    mid = np.array([(b0["x"] + b1["x"]) / 2, (b0["y"] + b1["y"]) / 2, 0.5])
    yaw = b0["yaw"]  # bins' back (handle side) normal
    back = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    side = np.array([-back[1], back[0], 0.0])
    cx, cy, cyaw = car["x"], car["y"], car["yaw"]
    fwd = np.array([math.cos(cyaw), math.sin(cyaw), 0.0])

    shots = {
        # Bins from behind (handle/wheel side), three-quarter view, eye height.
        "bins_back": (mid + 2.6 * back + 1.4 * side + [0, 0, 1.0], mid),
        # Bins from the street side.
        "bins_front": (mid - 3.0 * back - 1.0 * side + [0, 0, 1.4], mid),
        # Robot's-eye height, a few meters up the lane, looking at the bins.
        "approach_low": (mid + 6.0 * back + [0, 0, 0.25], mid + [0, 0, -0.1]),
        # At the garage looking down the driveway.
        "driveway": (np.array([cx, cy, 1.6]) - 2.0 * fwd, np.array([cx, cy, 0.0]) + 12.0 * fwd),
        # High three-quarter over the whole lot.
        "overview": (np.array([cx + 45.0, cy + 25.0, 45.0]), np.array([cx + 22.0, cy - 40.0, 0.0])),
    }
    for name, (eye, target) in shots.items():
        t0 = time.time()
        rgb = sim.render_view(eye, target, 1280, 720, fov=55.0, ss=2)
        Image.fromarray(rgb).save(f"{args.out}/{name}.png")
        print(f"{name}: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()

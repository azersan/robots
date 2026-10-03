"""Score recorded runs against ground truth: did the bin get home, and how much time was on grass?

    python tools/score_episodes.py [--url http://localhost:8642] [--last N]

Reads GET /api/episodes (episodes archived at each reset, plus the current one). For anna_pl,
"on pavement" means within the half-width of a road ribbon, using the same smoothed
centerlines the renderer draws. The grass check uses the robot's four chassis corners and
the four corners of the latched bin's footprint.
"""

import argparse
import math
import os
import sys

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simpg import site_anna_pl as site  # noqa: E402
from simpg.meshes import chaikin  # noqa: E402
from simpg.sim import BIN_TYPES  # noqa: E402

ROADS = [(chaikin([site.px_to_m(p) for p in r["px"]], 3), r["width"] / 2) for r in site.PAVEMENT]
CAR_CORNERS = [(0.3, 0.2), (0.3, -0.2), (-0.3, 0.2), (-0.3, -0.2)]
HOME_RADIUS = 3.0


def dist_to_polyline(p, pts):
    best = math.inf
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        dx, dy = bx - ax, by - ay
        t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy or 1.0)))
        best = min(best, math.hypot(p[0] - ax - t * dx, p[1] - ay - t * dy))
    return best


def off_pavement(p):
    return min(dist_to_polyline(p, pts) - hw for pts, hw in ROADS) > 0


def corners(x, y, yaw, pts):
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x + c * a - s * b, y + s * a + c * b) for a, b in pts]


def score(ep):
    cols = {c: i for i, c in enumerate(ep["columns"])}
    rows = ep["rows"]
    dt = ep.get("every_s", 0.1)
    sx, sy = ep["start"][:2]
    car_grass = 0.0
    dist = 0.0
    for r0, r1 in zip(rows, rows[1:]):
        dist += math.hypot(r1[cols["x"]] - r0[cols["x"]], r1[cols["y"]] - r0[cols["y"]])
    for r in rows:
        if any(off_pavement(p) for p in corners(r[cols["x"]], r[cols["y"]], r[cols["yaw"]], CAR_CORNERS)):
            car_grass += dt
    last = rows[-1]
    k = int(last[cols["latched"]])
    out = {"seed": ep["seed"], "sim_time": round(last[cols["t"]], 1), "distance_m": round(dist, 1),
           "robot_on_grass_s": round(car_grass, 1), "latched": k if k >= 0 else None}
    if k >= 0:
        bx, by, up = last[cols[f"bin{k}_x"]], last[cols[f"bin{k}_y"]], last[cols[f"bin{k}_upright"]]
        out["bin_to_start_m"] = round(math.hypot(bx - sx, by - sy), 2)
        out["bin_upright"] = round(up, 3)
        out["home"] = out["bin_to_start_m"] < HOME_RADIUS and up > 0.9
        # Bin footprint on grass while latched (approximate both carts by the larger one).
        t = max(BIN_TYPES.values(), key=lambda b: b["depth"])
        foot = [(sa * t["depth"] / 2, sb * t["width"] / 2) for sa in (-1, 1) for sb in (-1, 1)]
        out["bin_on_grass_s"] = round(sum(
            dt for r in rows if int(r[cols["latched"]]) == k and any(
                off_pavement(p) for p in corners(r[cols[f"bin{k}_x"]], r[cols[f"bin{k}_y"]], 0.0, foot))), 1)
    else:
        out["home"] = False
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8642")
    ap.add_argument("--last", type=int, default=10)
    args = ap.parse_args()
    d = httpx.get(f"{args.url}/api/episodes", timeout=30).json()
    eps = [e for e in d["archived"] if e["scene"] == "anna_pl"][-args.last:]
    for ep in eps:
        print(score(ep))


if __name__ == "__main__":
    main()

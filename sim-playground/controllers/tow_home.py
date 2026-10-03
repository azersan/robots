"""Drive the anna_pl route to the bins, latch onto one (found with the front depth camera),
and tow it back to the start by reversing up the route.

    python controllers/tow_home.py --seed 1 --trials 3

Once latched, the bin sits in front of the robot, so the robot drives *backwards* the whole
way home: the bin then trails behind the direction of travel like a towed trailer, which
is stable, and the robot never has to turn around with the bin attached.

Sensing: bins come from the front depth camera (perception.BinFinder); the car's own pose
comes from /api/state (GPS/odometry stand-in). True bin poses are used only for scoring.
"""

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import Gateway, lookahead_point, pure_pursuit, wrap  # noqa: E402
from perception import BinFinder  # noqa: E402

CF = 3                # sim frames per control tick (20 Hz)
DT = CF / 60.0
V_CRUISE = 1.2
V_DOCK = 0.22
V_TOW = 1.0           # reverse speed while towing
PREDOCK = 1.2
LOOKAHEAD = 1.2
TOW_LOOKAHEAD = 1.5
SEARCH_RADIUS = 12.0
FREEZE_RANGE = 0.9
HOME_TOL = 1.0        # stop once the estimated bin center is this close to the start


def heading_to(c, p):
    return math.atan2(p[1] - c["y"], p[0] - c["x"])


class BinTracker:
    """Track one bin's latch bar (world xy) and back-face yaw from front-camera depth."""

    def __init__(self, info, near, near_radius=6.0):
        self.finder = BinFinder(info)
        self.latch_out = info["bin"]["latch_out"]
        self.near, self.near_radius = near, near_radius  # bins are put out at the end of the route
        self.target = None
        self.cand, self.cand_hits = None, 0

    def update(self, gw, car, frozen=False):
        if self.target is not None and frozen:
            return self.target
        dets = self.finder.find(gw.depth("front"), car)
        if self.target is None:
            # Only accept bin-shaped things near the curb end of the route, seen 3 times in a row
            # at about the same place (trunks/shrubs sometimes pass the footprint test once).
            dets = [d for d in dets if math.dist(d["face_mid"], self.near) < self.near_radius]
            if not dets:
                self.cand_hits = 0
                return None
            pick = dets[0]
            if self.cand is not None and math.dist(pick["face_mid"], self.cand) < 0.5:
                self.cand_hits += 1
            else:
                self.cand_hits = 1
            self.cand = pick["face_mid"]
            if self.cand_hits < 3:
                return None
        elif not dets:
            return self.target
        else:
            lx, ly = self.target["latch"]
            pick = min(dets, key=lambda d: math.dist(d["face_mid"], (lx, ly)))
            if math.dist(pick["face_mid"], (lx, ly)) > 0.8:
                return self.target
        n = (math.cos(pick["yaw"]), math.sin(pick["yaw"]))
        new = {"latch": [pick["face_mid"][0] + n[0] * self.latch_out,
                         pick["face_mid"][1] + n[1] * self.latch_out],
               "yaw": pick["yaw"], "depth": pick["extents"]}
        if self.target is not None:
            a = 0.5
            new["latch"] = [a * new["latch"][i] + (1 - a) * self.target["latch"][i] for i in range(2)]
            new["yaw"] = self.target["yaw"] + a * wrap(new["yaw"] - self.target["yaw"])
        self.target = new
        return new


def dock_points(t, hook_x):
    lx, ly = t["latch"]
    nx, ny = math.cos(t["yaw"]), math.sin(t["yaw"])
    dock = (lx + nx * hook_x, ly + ny * hook_x)
    return (dock[0] + nx * PREDOCK, dock[1] + ny * PREDOCK), dock


class Run:
    def __init__(self, gw, info, verbose=1):
        self.gw, self.info, self.verbose = gw, info, verbose
        self.hook_x = info["car"]["hook_local"][0]
        self.route = [tuple(p) for p in info["route"]]
        self.st = None
        self.last_print = -10.0

    def tick(self, v, w, phase):
        self.gw.drive(v, w)
        self.st = self.gw.step(CF)
        if self.verbose and self.st["time"] - self.last_print >= 5.0:
            self.last_print = self.st["time"]
            c = self.st["car"]
            print(f"  t={self.st['time']:6.1f} {phase:8s} x={c['x']:6.2f} y={c['y']:6.2f} "
                  f"yaw={math.degrees(c['yaw']):5.0f} v={c['v']:.2f} latched={self.st['latched']}", flush=True)
        return self.st

    def check(self, timeout):
        if self.st["time"] > timeout:
            raise RuntimeError("timeout")
        if self.st["car"]["upright"] < 0.7:
            raise RuntimeError("robot_flipped")

    # ---- outbound: route -> find bin -> dock -> latch -------------------------------------
    def fetch(self, timeout):
        tracker = BinTracker(self.info, self.route[-1])
        c = self.st["car"]
        path = [(c["x"], c["y"])] + self.route[1:]
        target = None
        phase, retries, turned = "drive", 0, 0.0
        while True:
            self.check(timeout)
            c = self.st["car"]
            xy = (c["x"], c["y"])
            if phase != "drive" or math.dist(xy, path[-1]) < SEARCH_RADIUS:
                frozen = phase == "approach" and target is not None and \
                    math.dist(target["latch"], xy) < FREEZE_RANGE
                seen = tracker.update(self.gw, c, frozen)
                if seen is not None:
                    if target is None:
                        path = [xy, dock_points(seen, self.hook_x)[0]]
                        if phase == "search":
                            phase = "drive"
                    target = seen

            if phase == "drive":
                if target is not None:
                    path[-1] = dock_points(target, self.hook_x)[0]
                goal = path[-1]
                look, _, _ = lookahead_point(path, xy, LOOKAHEAD)
                dgoal = math.dist(xy, goal)
                v = V_CRUISE if dgoal > 3.0 else max(V_DOCK, V_CRUISE * dgoal / 3.0)
                v *= max(0.25, math.cos(wrap(heading_to(c, look) - c["yaw"])))
                w = pure_pursuit((c["x"], c["y"], c["yaw"]), look, v)
                if dgoal < 0.3:
                    phase = "align" if target is not None else "search"
            elif phase == "search":
                v, w = 0.0, 0.6
                turned += 0.6 * DT
                if turned > 2 * math.pi:
                    raise RuntimeError("bin_not_found")
            elif phase == "align":
                err = wrap(target["yaw"] + math.pi - c["yaw"])
                v, w = 0.0, max(-1.0, min(1.0, 2.5 * err))
                if abs(err) < math.radians(3):
                    phase = "approach"
            else:  # approach / backoff along the dock line, Stanley-style
                _, dock = dock_points(target, self.hook_x)
                ly = wrap(target["yaw"] + math.pi)
                ux, uy = math.cos(ly), math.sin(ly)
                dx, dy = c["x"] - dock[0], c["y"] - dock[1]
                togo = -(dx * ux + dy * uy)
                cross = -dx * uy + dy * ux
                herr = wrap(ly - c["yaw"])
                if phase == "backoff":
                    v, w = -0.2, 1.5 * herr
                    if togo > PREDOCK * 0.8:
                        phase, retries = "approach", retries + 1
                else:
                    v = max(0.05, min(V_DOCK, 0.6 * togo))
                    w = 2.0 * herr - math.atan2(2.0 * cross, v + 0.1)
                    h2l = math.dist(c["hook"][:2], target["latch"])
                    if togo < 0.15 and abs(cross) > 0.04:
                        if retries >= 3:
                            raise RuntimeError("dock_missed")
                        phase = "backoff"
                    elif h2l < 0.03 or togo < 0.0:
                        self.gw.drive(0.0, 0.0)
                        self.st = self.gw.step(12)
                        r = self.gw.latch(True)
                        if r["latched"] is not None:
                            self.st = self.gw.state()
                            return target
                        if retries >= 3:
                            raise RuntimeError(f"latch_failed d={r['nearest']['distance']:.3f}")
                        phase, retries = "backoff", retries + 1
                        continue
            self.tick(v, w, phase)

    # ---- inbound: reverse up the route with the bin in tow ------------------------------
    def tow_home(self, start, target, bin_half, timeout):
        c = self.st["car"]
        xy = (c["x"], c["y"])
        # First back straight out along the dock line to the pre-dock point, then join the
        # route (skipping route points that lie back toward the bins) and run it in reverse.
        back = (xy[0] + math.cos(c["yaw"] + math.pi) * PREDOCK, xy[1] + math.sin(c["yaw"] + math.pi) * PREDOCK)
        rev = self.route[::-1]
        k = 0
        while k < len(rev) - 1 and math.dist(rev[k], back) < 4.0:
            k += 1
        path = [xy, back] + rev[k:-1] + [start]
        nose_to_center = self.hook_x + self.info["bin"]["latch_out"] + bin_half
        while True:
            self.check(timeout)
            c = self.st["car"]
            xy = (c["x"], c["y"])
            yaw_b = wrap(c["yaw"] + math.pi)       # direction of travel
            bin_c = (c["x"] + math.cos(c["yaw"]) * nose_to_center, c["y"] + math.sin(c["yaw"]) * nose_to_center)
            if self.st["latched"] is None:
                raise RuntimeError("unlatched")
            if math.dist(bin_c, start) < HOME_TOL or math.dist(xy, start) < 0.25:
                break
            look, _, _ = lookahead_point(path, xy, TOW_LOOKAHEAD)
            dgoal = math.dist(xy, start)
            s = V_TOW if dgoal > 3.0 else max(0.3, V_TOW * dgoal / 3.0)
            herr = wrap(heading_to(c, look) - yaw_b)
            if abs(herr) > math.radians(70):
                v, w = 0.0, max(-0.6, min(0.6, 1.5 * herr))  # gentle pivot; the bin swings around
            else:
                s *= max(0.3, math.cos(herr))
                w = pure_pursuit((c["x"], c["y"], yaw_b), look, s)
                w = max(-1.0, min(1.0, w))
                v = -s
            self.tick(v, w, "tow")
        for _ in range(20):  # come to a stop, keep the watchdog fed
            self.tick(0.0, 0.0, "stop")


def run_trial(gw, seed, randomize=True, verbose=1, timeout=600.0):
    st = gw.reset(scene="anna_pl", randomize=randomize, seed=seed)
    info = gw.info()
    gw.lockstep()
    gw.post("/api/config", cmd_timeout=1.0)
    start = (st["car"]["x"], st["car"]["y"])
    run = Run(gw, info, verbose)
    run.st = st
    t0 = time.time()
    res = {"seed": seed}
    try:
        target = run.fetch(timeout)
        k = run.st["latched"]
        res["latched_at"] = round(run.st["time"], 1)
        kind = run.st["bins"][k]["kind"]  # scoring/debug only: which type, for the depth offset
        # The tracker could also estimate this; use the mean of the two cart depths to stay sensor-only.
        types = info["bin"]["types"].values()
        half = sum(t["depth"] for t in types) / len(types) / 2
        run.tow_home(start, target, half, timeout)
        res["result"] = "home"
        res["kind"] = kind
    except RuntimeError as e:
        res["result"] = str(e)
    st = gw.state()
    res["sim_time"] = round(st["time"], 1)
    res["wall"] = round(time.time() - t0, 1)
    k = st["latched"]
    res["latched"] = k
    if k is not None:
        b = st["bins"][k]
        res["bin_dist"] = round(math.dist((b["x"], b["y"]), start), 2)
        res["bin_upright"] = round(b["upright"], 3)
        res["success"] = res["bin_dist"] < 3.0 and b["upright"] > 0.9
    else:
        res["success"] = False
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8642")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--no-randomize", action="store_true")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()
    gw = Gateway(args.url)
    results = []
    try:
        for i in range(args.trials):
            seed = args.seed + i
            print(f"trial seed={seed}", flush=True)
            r = run_trial(gw, seed, not args.no_randomize, verbose=0 if args.quiet else 1)
            print("  ->", r, flush=True)
            results.append(r)
    finally:
        gw.drive(0.0, 0.0)
        gw.post("/api/config", cmd_timeout=1.0)
        gw.realtime()
    print(f"\nsuccess {sum(r['success'] for r in results)}/{len(results)}")


if __name__ == "__main__":
    main()

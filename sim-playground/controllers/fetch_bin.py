"""Drive the route, find a bin, dock on its latch bar, latch, and tug it back a meter.

    python controllers/fetch_bin.py --scene anna_pl --sense camera --trials 5 --seed 1

--sense truth   bin poses come from the sim (rung 1)
--sense camera  bins are found in the front depth camera (rung 2); the car's own pose
                still comes from the sim, standing in for GPS/odometry

Runs the gateway in lockstep (20 Hz control, 3 sim frames per command) so it goes as
fast as the sim can and is repeatable for a given seed. Scoring uses the true bin pose.
"""

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from common import Gateway, closest_index, lookahead_point, pure_pursuit, wrap  # noqa: E402
from perception import BinFinder  # noqa: E402

CONTROL_FRAMES = 3  # 60 fps sim / 3 = 20 Hz control
V_CRUISE = 1.2
V_DOCK = 0.25
PREDOCK = 1.2  # m in front of the latch bar to line up before the final approach
LOOKAHEAD = 1.2
SEARCH_RADIUS = 12.0  # start looking for bins this close to the end of the route
FREEZE_RANGE = 0.9  # stop updating the camera estimate this close (face fills the view)


def dock_points(target, hook_offset):
    """Pre-dock and dock car positions in front of a target {"latch": [x, y, ...], "yaw": front normal}."""
    lx, ly = target["latch"][:2]
    nx, ny = math.cos(target["yaw"]), math.sin(target["yaw"])
    dock = (lx + nx * hook_offset, ly + ny * hook_offset)
    pre = (dock[0] + nx * PREDOCK, dock[1] + ny * PREDOCK)
    return pre, dock


def build_path(route, goal, car_xy):
    """Route up to where it passes closest to the goal, then the goal itself."""
    if len(route) >= 2:
        cut = closest_index(route, goal)
        return [car_xy] + list(route[1:cut]) + [goal]
    return [car_xy, goal]


class TruthSensor:
    def __init__(self, bin_id):
        self.bin_id = bin_id

    def update(self, gw, st, phase):
        b = st["bins"][self.bin_id]
        return {"latch": b["latch"], "yaw": b["yaw"]}


class CameraSensor:
    """Tracks one bin seen by the depth camera. Returns None until it has seen one."""

    def __init__(self, info):
        self.finder = BinFinder(info)
        self.latch_out = info["bin"]["latch_out"]  # latch bar sits this far out from the face we dock on
        self.target = None
        self.sightings = 0

    def update(self, gw, st, phase):
        car = st["car"]
        if self.target and phase == "approach":
            lx, ly = self.target["latch"][:2]
            if math.hypot(lx - car["x"], ly - car["y"]) < FREEZE_RANGE:
                return self.target
        dets = self.finder.find(gw.depth("front"), car)
        if not dets:
            return self.target
        if self.target is None:
            pick = dets[0]  # closest
        else:
            lx, ly = self.target["latch"][:2]
            pick = min(dets, key=lambda d: math.dist(d["face_mid"], (lx, ly)))
            if math.dist(pick["face_mid"], (lx, ly)) > 0.8:
                return self.target  # lost it in clutter; keep the old estimate
        n = (math.cos(pick["yaw"]), math.sin(pick["yaw"]))
        new = {"latch": [pick["face_mid"][0] + n[0] * self.latch_out, pick["face_mid"][1] + n[1] * self.latch_out],
               "yaw": pick["yaw"]}
        if self.target is not None and phase != "align":
            # Light smoothing; yaw blended on the circle.
            a = 0.5
            new["latch"] = [a * new["latch"][i] + (1 - a) * self.target["latch"][i] for i in range(2)]
            new["yaw"] = self.target["yaw"] + a * wrap(new["yaw"] - self.target["yaw"])
        self.target = new
        self.sightings += 1
        return self.target


def run_trial(gw, scene, sense, bin_id, randomize, seed, verbose=True, timeout=400.0):
    st = gw.reset(scene=scene, randomize=randomize, seed=seed)
    info = gw.info()
    gw.lockstep()
    gw.post("/api/config", cmd_timeout=0.5)
    hook_offset = info["car"]["hook_local"][0]
    route = [tuple(p) for p in info["route"]]
    sensor = TruthSensor(bin_id) if sense == "truth" else CameraSensor(info)

    target = sensor.update(gw, st, "drive") if sense == "truth" else None
    goal = dock_points(target, hook_offset)[0] if target else route[-1]
    path = build_path(route, goal, (st["car"]["x"], st["car"]["y"]))
    t_wall = time.time()
    phase, retries, search_turned, last_print = "drive", 0, 0.0, -10.0

    while True:
        c = st["car"]
        pose = (c["x"], c["y"], c["yaw"])
        if st["time"] > timeout:
            phase = "timeout"
            break
        if c["upright"] < 0.7:
            phase = "flipped"
            break

        near_end = math.dist((c["x"], c["y"]), path[-1]) < SEARCH_RADIUS
        if sense == "truth" or near_end or phase != "drive":
            seen = sensor.update(gw, st, phase)
            if seen is not None:
                if target is None:  # first sighting: re-plan to its pre-dock point
                    path = [(c["x"], c["y"]), dock_points(seen, hook_offset)[0]]
                    if phase == "search":
                        phase = "drive"
                target = seen

        if phase == "drive":
            if target is not None:
                pre = dock_points(target, hook_offset)[0]
                path[-1] = pre  # follow the live estimate
            goal = path[-1]
            look, _, _ = lookahead_point(path, (c["x"], c["y"]), LOOKAHEAD)
            to_goal = math.dist((c["x"], c["y"]), goal)
            v = V_CRUISE if to_goal > 3.0 else max(V_DOCK, V_CRUISE * to_goal / 3.0)
            heading_err = wrap(math.atan2(look[1] - c["y"], look[0] - c["x"]) - c["yaw"])
            v *= max(0.25, math.cos(heading_err))
            w = pure_pursuit(pose, look, v)
            if to_goal < 0.3:
                phase = "align" if target is not None else "search"
        elif phase == "search":
            # End of the route and nothing seen yet: turn in place and look around.
            v, w = 0.0, 0.6
            search_turned += 0.6 * CONTROL_FRAMES / 60
            if search_turned > 2 * math.pi:
                phase = "not_found"
                break
        elif phase == "align":
            want = wrap(target["yaw"] + math.pi)  # face the bin
            err = wrap(want - c["yaw"])
            v, w = 0.0, max(-1.0, min(1.0, 2.5 * err))
            if abs(err) < math.radians(3):
                phase = "approach"
        elif phase in ("approach", "backoff"):
            # Follow the dock line (through the dock point, pointing into the bin) with Stanley steering.
            _, dock = dock_points(target, hook_offset)
            line_yaw = wrap(target["yaw"] + math.pi)
            ux, uy = math.cos(line_yaw), math.sin(line_yaw)
            dx, dy = c["x"] - dock[0], c["y"] - dock[1]
            dist_to_go = -(dx * ux + dy * uy)  # along the line, > 0 before the dock point
            cross = -dx * uy + dy * ux  # + means car is left of the line
            heading_err = wrap(line_yaw - c["yaw"])
            if phase == "backoff":
                v, w = -0.2, 1.5 * heading_err
                if dist_to_go > PREDOCK * 0.8:
                    phase, retries = "approach", retries + 1
            else:
                v = max(0.05, min(V_DOCK, 0.6 * dist_to_go))
                w = 2.0 * heading_err - math.atan2(2.0 * cross, v + 0.1)
                hook_to_latch = math.dist(c["hook"][:2], target["latch"][:2])
                if dist_to_go < 0.15 and abs(cross) > 0.04:
                    if retries >= 3:
                        phase = "missed"
                        break
                    phase = "backoff"
                elif hook_to_latch < 0.03 or dist_to_go < 0.0:
                    gw.drive(0.0, 0.0)
                    st = gw.step(10)
                    r = gw.latch(True)
                    phase = "latched" if r["latched"] is not None else "missed"
                    break
        gw.drive(v, w)
        st = gw.step(CONTROL_FRAMES)
        if verbose and (st["time"] - last_print >= 5.0 or (verbose == 2 and phase != "drive")):
            last_print = st["time"]
            tgt = "target=?"
            if target:
                err = min(math.dist(target["latch"][:2], b["latch"][:2]) for b in st["bins"])
                yerr = min(abs(math.degrees(wrap(target["yaw"] - b["yaw"]))) for b in st["bins"])
                tgt = f"target=({target['latch'][0]:.2f},{target['latch'][1]:.2f}) err={err:.2f}m yaw_err={yerr:.0f}deg"
            print(f"  t={st['time']:6.1f}s {phase:8s} x={c['x']:6.1f} y={c['y']:6.1f} v={c['v']:.2f} {tgt}")

    st = gw.state()
    result = {"seed": seed, "result": phase, "sim_time": round(st["time"], 1),
              "wall": round(time.time() - t_wall, 1), "retries": retries}
    if target is not None:
        # How far the estimate was from the nearest true latch bar.
        result["est_err"] = round(min(math.dist(target["latch"][:2], b["latch"][:2]) for b in st["bins"]), 3)
    if phase == "latched":
        k = st["latched"]
        result["bin"] = k
        before = st["bins"][k]
        for _ in range(60):  # back up for 3 s, re-sending so the watchdog stays fed
            gw.drive(-0.3, 0.0)
            gw.step(CONTROL_FRAMES)
        gw.drive(0.0, 0.0)
        st = gw.step(30)
        b = st["bins"][k]
        result["bin_moved"] = round(math.dist((b["x"], b["y"]), (before["x"], before["y"])), 2)
        result["bin_upright"] = round(b["upright"], 3)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8642")
    ap.add_argument("--scene", default="flat")
    ap.add_argument("--sense", choices=["truth", "camera"], default="camera")
    ap.add_argument("--bin", type=int, default=0, help="which bin to fetch with --sense truth")
    ap.add_argument("--randomize", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--realtime", action="store_true", help="leave the gateway running in real time afterwards")
    ap.add_argument("-v", "--verbose", action="count", default=1, help="-v logs every tick after the drive phase")
    args = ap.parse_args()

    gw = Gateway(args.url)
    results = []
    for k in range(args.trials):
        seed = args.seed + k
        print(f"trial {k + 1}/{args.trials} scene={args.scene} sense={args.sense} seed={seed}")
        r = run_trial(gw, args.scene, args.sense, args.bin, args.randomize or args.trials > 1, seed,
                      verbose=args.verbose)
        print("  ->", r)
        results.append(r)
    ok = sum(r["result"] == "latched" for r in results)
    print(f"\nlatched {ok}/{len(results)}")
    gw.post("/api/config", cmd_timeout=1.0)
    if args.realtime:
        gw.realtime()


if __name__ == "__main__":
    main()

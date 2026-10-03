"""Drive a small differential-drive car up a scanned driveway in Newton (MuJoCo Warp on the GPU).

The scene comes from scan2sim (<scan>.scan/sim/). The car starts at the scan origin (where you pressed Start)
and follows the path you walked on the way out, using pure pursuit.

    python sim/driveway.py ~/scans/driveway-2026-10-03-1746.scan               # web viewer on :8080
    python sim/driveway.py ~/scans/driveway-2026-10-03-1746.scan --viewer none # headless, prints progress

From the Mac: ssh -N -L 8080:127.0.0.1:8080 nzxt   then open http://localhost:8080
(127.0.0.1, not localhost: Windows resolves localhost to IPv6, which WSL does not forward.)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton

# Car geometry (meters, kg): a small rover. Body frame: +x forward, +y left, +z up.
# 10 cm wheels get over driveway lips of a few cm; nothing this size climbs a 15 cm curb.
CHASSIS_HALF = (0.25, 0.15, 0.05)
CHASSIS_MASS = 4.0
WHEEL_RADIUS = 0.10
WHEEL_HALF_WIDTH = 0.025
WHEEL_MASS = 0.4
TRACK = 0.40  # distance between wheel centers
WHEEL_X = -0.08  # wheels behind center, caster in front
CASTER_RADIUS = 0.04
CASTER_X = 0.20
WHEEL_TORQUE = 1.5  # N·m per wheel: 15 N of push each, well under the tip-over torque of the chassis

# Path following.
SPEED = 0.6  # m/s
LOOKAHEAD = 1.0  # m
GOAL_TOLERANCE = 0.5  # m


def load_heightfield(sim: Path):
    rep = json.loads((sim / "report.json").read_text())["heightfield"]
    raw = np.fromfile(sim / rep["file"], dtype="<f4")
    ny, nx = raw[:2].view("<i4")
    z = raw[2:].reshape(ny, nx).astype(np.float32)
    return rep, z


def ground_height(rep, z, x, y):
    ny, nx = z.shape
    c = int(np.clip(round((x - rep["x0"]) / rep["cell"]), 0, nx - 1))
    r = int(np.clip(round((y - rep["y0"]) / rep["cell"]), 0, ny - 1))
    return float(z[r, c])


def walked_path(scan: Path, spacing: float = 0.5) -> np.ndarray:
    """XY waypoints of the outbound walk (keyframe camera positions up to the farthest point), smoothed."""
    poses = json.loads((scan / "poses.json").read_text())["keyframes"]
    pos = np.array([np.array(k["camera_to_world"]).reshape(4, 4)[:2, 3] for k in poses])
    d = np.linalg.norm(pos - pos[0], axis=1)
    far = int(np.argmax(d))
    # Skip the look-around at the start: begin at the last moment before setting off that you were near the start.
    begin = int(np.nonzero(d[: far + 1] < 1.0)[0].max())
    pos = pos[begin : far + 1]
    # Moving-average smooth, then resample by arc length.
    k = 7
    padded = np.vstack([np.repeat(pos[:1], k, 0), pos, np.repeat(pos[-1:], k, 0)])
    smooth = np.array([padded[i : i + 2 * k + 1].mean(axis=0) for i in range(len(pos))])
    seg = np.linalg.norm(np.diff(smooth, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    samples = np.arange(0, s[-1], spacing)
    return np.column_stack([np.interp(samples, s, smooth[:, 0]), np.interp(samples, s, smooth[:, 1])])


def build(scan: Path, start_xy: np.ndarray, heading: float):
    sim = scan / "sim"
    builder = newton.ModelBuilder()
    # Visual meshes from the scan; the mesh colliders are replaced by the exact heightfield below
    # (MuJoCo would collide with a mesh's convex hull).
    builder.add_usd(str(sim / "scene.usda"), ignore_paths=[".*/Collision.*"])

    rep, z = load_heightfield(sim)
    # 10 cm collision cells: at 5 cm a 30 cm car touches too many cells for MuJoCo's contact budget.
    z, rep = z[::2, ::2].copy(), {**rep, "cell": rep["cell"] * 2}
    ny, nx = z.shape
    hx, hy = (nx - 1) * rep["cell"] / 2, (ny - 1) * rep["cell"] / 2
    hf = newton.Heightfield(data=z, nrow=ny, ncol=nx, hx=hx, hy=hy, min_z=float(z.min()), max_z=float(z.max()))
    builder.add_shape_heightfield(xform=wp.transform(wp.vec3(rep["x0"] + hx, rep["y0"] + hy, 0.0), wp.quat_identity()),
                                  heightfield=hf, label="ground")

    gz = ground_height(rep, z, *start_xy)  # same 10 cm grid the car collides with
    q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), heading)
    root = wp.transform(wp.vec3(float(start_xy[0]), float(start_xy[1]), gz + WHEEL_RADIUS + 0.02), q)
    chassis = add_car(builder, root)
    return builder.finalize(), chassis


def add_car(builder: newton.ModelBuilder, root: wp.transform) -> int:
    """Differential-drive car: box chassis, two velocity-controlled wheels, a slick front caster. Returns the
    chassis body. `root` is the axle-height pose of the chassis center."""
    chassis = builder.add_link(xform=root, label="car")
    builder.add_shape_box(chassis, hx=CHASSIS_HALF[0], hy=CHASSIS_HALF[1], hz=CHASSIS_HALF[2],
                          cfg=newton.ModelBuilder.ShapeConfig(density=float(CHASSIS_MASS / (8 * np.prod(CHASSIS_HALF)))),
                          color=wp.vec3(0.9, 0.3, 0.1))
    slick = newton.ModelBuilder.ShapeConfig(mu=0.01, density=200.0)  # near-frictionless caster
    builder.add_shape_sphere(chassis, xform=wp.transform(wp.vec3(CASTER_X, 0.0, -(WHEEL_RADIUS - CASTER_RADIUS)),
                                                         wp.quat_identity()), radius=CASTER_RADIUS, cfg=slick)
    joints = [builder.add_joint_free(chassis)]

    # Cylinders are Z-aligned; rotate so the axle runs along body Y.
    axle = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi / 2)
    wheel_volume = math.pi * WHEEL_RADIUS**2 * 2 * WHEEL_HALF_WIDTH
    for side, name in [(1, "left"), (-1, "right")]:
        offset = wp.vec3(WHEEL_X, side * TRACK / 2, 0.0)
        wheel = builder.add_link(xform=root * wp.transform(offset, wp.quat_identity()), label=f"wheel_{name}")
        builder.add_shape_cylinder(wheel, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), axle), radius=WHEEL_RADIUS,
                                   half_height=WHEEL_HALF_WIDTH,
                                   cfg=newton.ModelBuilder.ShapeConfig(mu=1.2, density=float(WHEEL_MASS / wheel_volume)),
                                   color=wp.vec3(0.1, 0.1, 0.1))
        joints.append(builder.add_joint_revolute(
            chassis, wheel, parent_xform=wp.transform(offset, wp.quat_identity()), axis=(0.0, 1.0, 0.0),
            actuator_mode=newton.JointTargetMode.VELOCITY, target_vel=0.0, target_kd=1.0, effort_limit=WHEEL_TORQUE,
            label=f"axle_{name}"))
    builder.add_articulation(joints, label="car")
    return chassis


class PurePursuit:
    def __init__(self, path: np.ndarray):
        self.path = path
        self.i = 0

    def command(self, x: float, y: float, yaw: float) -> tuple[float, float, bool]:
        """Returns (forward speed m/s, yaw rate rad/s, done)."""
        p = np.array([x, y])
        if np.linalg.norm(self.path[-1] - p) < GOAL_TOLERANCE:
            return 0.0, 0.0, True
        # Advance the index to the closest point, then look ahead.
        while self.i + 1 < len(self.path) and np.linalg.norm(self.path[self.i + 1] - p) <= np.linalg.norm(self.path[self.i] - p):
            self.i += 1
        j = self.i
        while j + 1 < len(self.path) and np.linalg.norm(self.path[j] - p) < LOOKAHEAD:
            j += 1
        dx, dy = self.path[j] - p
        alpha = math.atan2(dy, dx) - yaw
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))
        v = SPEED * max(0.3, math.cos(alpha))  # slow down for sharp turns
        return v, 2 * v * math.sin(alpha) / LOOKAHEAD, False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan", type=Path, help="<name>.scan folder containing sim/ and poses.json")
    ap.add_argument("--viewer", choices=["viser", "gl", "none"], default="viser")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--seconds", type=float, default=180.0, help="max sim time (per run)")
    ap.add_argument("--loop", action="store_true", help="restart from the street each time the car finishes (viewer demo)")
    args = ap.parse_args()

    path = walked_path(args.scan)
    heading = math.atan2(*(path[min(4, len(path) - 1)] - path[0])[::-1])
    model, chassis = build(args.scan, path[0], heading)
    # MuJoCo Warp's own heightfield collision puts the surface ~10-15 cm too high on this terrain (Newton 1.6 /
    # mujoco-warp 3.12); Newton's collision pipeline is exact, so it supplies the contacts to the MuJoCo solver.
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=False)
    collision = newton.CollisionPipeline(model)
    contacts = collision.contacts()
    s0, s1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    start_q, start_qd = wp.clone(s0.joint_q), wp.clone(s0.joint_qd)

    def reset():
        # Write into the same buffers so the captured CUDA graph stays valid.
        for st in (s0, s1):
            wp.copy(st.joint_q, start_q)
            wp.copy(st.joint_qd, start_qd)
            newton.eval_fk(model, st.joint_q, st.joint_qd, st)

    viewer = None
    if args.viewer == "viser":
        viewer = newton.viewer.ViewerViser(port=args.port)
    elif args.viewer == "gl":
        viewer = newton.viewer.ViewerGL()
    if viewer:
        viewer.set_model(model)
        # Start behind and above the car, looking up the driveway (yaw in degrees about +Z, 0 = looking along +X).
        back = path[0] - 4.0 * np.array([math.cos(heading), math.sin(heading)])
        viewer.set_camera(pos=wp.vec3(float(back[0]), float(back[1]), 2.5), pitch=-20.0, yaw=math.degrees(heading))

    fps, substeps = 60, 4  # even, so the s0/s1 swaps inside the captured graph end on the same buffers
    frame_dt = 1.0 / fps
    dt = frame_dt / substeps
    pursuit = PurePursuit(path)
    qd_start = model.joint_qd_start.numpy()
    axle_dofs = [int(qd_start[j]) for j in range(model.joint_count) if model.joint_label[j].startswith("axle_")]
    target = np.zeros(model.joint_dof_count, dtype=np.float32)
    t, last_print, done, wall0 = 0.0, -10.0, False, time.time()
    progress_t, progress_i = 0.0, 0
    print(f"Path: {len(path)} waypoints, {np.linalg.norm(np.diff(path, axis=0), axis=1).sum():.1f} m")

    states = [s0, s1]

    def simulate():
        a, b = states
        for _ in range(substeps):
            a.clear_forces()
            collision.collide(a, contacts)
            solver.step(a, b, control, contacts, dt)
            a, b = b, a

    graph = None
    while (args.loop or t < args.seconds) and (viewer is None or viewer.is_running()):
        body = s0.body_q.numpy()[chassis]
        x, y, zc = body[:3]
        qx, qy, qz, qw = body[3:]
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (qx * qx + qy * qy)))))  # body z vs world z
        v, w, done = pursuit.command(x, y, yaw)
        left, right = (v - w * TRACK / 2) / WHEEL_RADIUS, (v + w * TRACK / 2) / WHEEL_RADIUS
        target[axle_dofs] = [left, right]
        control.joint_target_qd.assign(target)

        if graph is None and wp.get_device().is_cuda and t > 0:
            # Record one frame of substeps once, then replay it: single-robot sims are launch-bound otherwise.
            with wp.ScopedCapture() as capture:
                simulate()
            graph = capture.graph
        elif graph is not None:
            wp.capture_launch(graph)
        else:
            simulate()
        t += frame_dt

        if viewer:
            viewer.begin_frame(t)
            viewer.log_state(s0)
            viewer.end_frame()
        if t - last_print >= 5.0 or done:
            last_print = t
            print(f"t={t:6.1f}s  car at ({x:6.2f}, {y:6.2f}, z={zc:5.2f}) tilt {tilt:4.0f} deg  waypoint {pursuit.i}/{len(path) - 1}  "
                  f"(sim {t / max(time.time() - wall0, 1e-6):.1f}x real time)")
        if pursuit.i > progress_i:
            progress_t, progress_i = t, pursuit.i
        stuck = t - progress_t > 15.0
        if stuck:
            print(f"Stuck near ({x:.2f}, {y:.2f}) for 15 s (waypoint {pursuit.i}).")
        if done:
            print(f"Reached the end of the walked path in {t:.1f} s of sim time.")
        if done or stuck or t >= args.seconds:
            if not args.loop:
                break
            reset()
            pursuit = PurePursuit(path)
            t, last_print, progress_t, progress_i, done = 0.0, -10.0, 0.0, 0, False
            wall0 = time.time()
            print("Restarting from the street.")
    if viewer and done:
        print("Viewer stays up; Ctrl-C to quit.")
        while viewer.is_running():
            viewer.begin_frame(t)
            viewer.log_state(s0)
            viewer.end_frame()
            time.sleep(1 / 30)


if __name__ == "__main__":
    main()

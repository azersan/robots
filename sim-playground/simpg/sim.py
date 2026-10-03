"""Newton-backed simulator for the driveway car.

World frame: Z up, X east, Y north, meters. The car's body frame is X forward,
Y left, Z up. Everything here runs on one thread at a time; the gateway holds
a lock around every call.
"""

import math
import random
from dataclasses import dataclass, field

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorTiledCamera

from . import site_anna_pl as site

wp.config.quiet = True

# ---------------------------------------------------------------------------
# Dimensions. The car is a small differential-drive tug with a caster in front
# and a tow hook on its nose; bins are 96-gallon carts.

WHEEL_RADIUS = 0.10
WHEEL_HALF_WIDTH = 0.025
TRACK = 0.48  # wheel center to wheel center
CHASSIS_HALF = (0.30, 0.20, 0.06)
WHEEL_MOUNT = (-0.10, -0.02)  # (x, z) of the axle in the chassis frame
CASTER_RADIUS = 0.04
CHASSIS_Z0 = WHEEL_RADIUS - WHEEL_MOUNT[1]
HOOK_LOCAL = (CHASSIS_HALF[0] + 0.04, 0.0, 0.08 - CHASSIS_Z0)  # 8 cm off the ground

MAX_WHEEL_SPEED = 2.0  # m/s at the tire
WHEEL_EFFORT = 20.0  # N*m per wheel

BIN_HALF = (0.37, 0.31, 0.53)  # depth, width, height / 2
BIN_MASS = 13.0
LATCH_LOCAL = (BIN_HALF[0] + 0.03, 0.0, 0.08 - BIN_HALF[2])  # handle bar low on the front face
LATCH_RANGE = 0.08  # hook must be this close to a latch bar to engage

HITCH_K = 4000.0
HITCH_C = 150.0

FPS = 60
SUBSTEPS = 10

# Contact friction is the max of the two surfaces, so the ground stays low and each
# object's own mu (tires 1.0, bin 0.25, ...) decides.
GROUND_CFG = newton.ModelBuilder.ShapeConfig(mu=0.2, gap=0.01)

GRASS = (0.33, 0.47, 0.22)
ASPHALT = (0.36, 0.36, 0.38)
SKY = 0xFFEBC396  # packed 0xAABBGGRR

# Front camera mount in the chassis frame: position and look direction (pitched down ~7 deg).
FRONT_CAM_POS = (CHASSIS_HALF[0] + 0.01, 0.0, CHASSIS_HALF[2] + 0.06)
FRONT_CAM_DIR = (1.0, 0.0, -0.12)

CAMERAS = {
    # name: (width, height, vertical fov deg)
    "front": (320, 240, 70.0),
    "chase": (480, 320, 60.0),
    "overhead": (512, 512, 50.0),
}


def _visual(cfg=None):
    c = cfg or newton.ModelBuilder.ShapeConfig()
    c.has_shape_collision = False
    c.has_particle_collision = False
    c.collision_group = 0
    c.density = 0.0
    return c


def _quat_z(yaw):
    return wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(yaw))


def _yaw_of(q):
    qx, qy, qz, qw = q
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _rotate(q, v):
    qx, qy, qz, qw = q
    u = np.array([qx, qy, qz])
    v = np.asarray(v, dtype=float)
    return v + 2.0 * np.cross(u, np.cross(u, v) + qw * v)


def look_at_quat(eye, target, up=(0.0, 0.0, 1.0)):
    """Camera-to-world rotation (x, y, z, w) for an OpenGL-style camera (looks down -Z, +Y up)."""
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-6:  # looking straight down: pick north as image-up
        r = np.cross(f, (0.0, 1.0, 0.0))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.stack([r, u, -f], axis=1)
    return _mat_to_quat(m)


def _mat_to_quat(m):
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return ((m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s)
    i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
    if i == 0:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        return (0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s)
    if i == 1:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        return ((m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s)
    s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
    return ((m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s)


# ---------------------------------------------------------------------------
# Scene construction


@dataclass
class SceneSpec:
    name: str
    start: tuple  # (x, y, yaw)
    bins: list  # [(x, y, yaw, color)]
    route: list = field(default_factory=list)  # [(x, y)] reference path, if any
    bounds: tuple = (-10.0, -10.0, 10.0, 10.0)  # xmin, ymin, xmax, ymax for the overhead view


def add_car(builder, x, y, yaw):
    chassis = builder.add_link(
        xform=wp.transform(p=wp.vec3(x, y, CHASSIS_Z0), q=_quat_z(yaw)), label="car"
    )
    # ~17 kg chassis: a real tug carries its battery low for traction.
    body_cfg = newton.ModelBuilder.ShapeConfig(density=600.0, mu=0.6, gap=0.01)
    hx, hy, hz = CHASSIS_HALF
    builder.add_shape_box(chassis, hx=hx, hy=hy, hz=hz, cfg=body_cfg, color=(0.85, 0.45, 0.10), label="car_body")
    # Camera mast and tow hook, for looks.
    builder.add_shape_box(
        chassis, xform=wp.transform(p=wp.vec3(hx - 0.04, 0.0, hz + 0.04)), hx=0.03, hy=0.05, hz=0.04,
        cfg=_visual(), color=(0.1, 0.1, 0.1), label="car_camera",
    )
    builder.add_shape_box(
        chassis, xform=wp.transform(p=wp.vec3(*HOOK_LOCAL)), hx=0.04, hy=0.06, hz=0.015,
        cfg=_visual(), color=(0.75, 0.75, 0.78), label="car_hook",
    )
    joints = [builder.add_joint_free(chassis, label="car_free")]

    # Front caster: a ball on a ball joint, so it rolls freely in any direction.
    caster = builder.add_link(label="car_caster")
    builder.add_shape_sphere(
        caster, radius=CASTER_RADIUS, cfg=newton.ModelBuilder.ShapeConfig(density=500.0, mu=0.6, gap=0.01),
        color=(0.2, 0.2, 0.2), label="car_caster",
    )
    joints.append(builder.add_joint_ball(
        parent=chassis, child=caster,
        parent_xform=wp.transform(p=wp.vec3(hx - 0.08, 0.0, -(CHASSIS_Z0 - CASTER_RADIUS))),
        child_xform=wp.transform(), label="car_caster",
    ))

    wheel_cfg = newton.ModelBuilder.ShapeConfig(density=400.0, mu=1.0, gap=0.01)
    drive = []
    for side, sy in (("left", 1.0), ("right", -1.0)):
        wheel = builder.add_link(label=f"car_wheel_{side}")
        builder.add_shape_cylinder(
            wheel, xform=wp.transform(q=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.5 * math.pi)),
            radius=WHEEL_RADIUS, half_height=WHEEL_HALF_WIDTH, cfg=wheel_cfg, color=(0.08, 0.08, 0.09),
            label=f"car_wheel_{side}",
        )
        j = builder.add_joint_revolute(
            parent=chassis, child=wheel,
            parent_xform=wp.transform(p=wp.vec3(WHEEL_MOUNT[0], sy * TRACK / 2, WHEEL_MOUNT[1])),
            child_xform=wp.transform(), axis=newton.Axis.Y,
            target_vel=0.0, target_kd=30.0, damping=0.05, armature=0.01,
            effort_limit=WHEEL_EFFORT, actuator_mode=newton.JointTargetMode.VELOCITY,
            label=f"car_drive_{side}",
        )
        drive.append(j)
        joints.append(j)
    builder.add_articulation(joints, label="car")
    return chassis, joints[0], drive


def add_bin(builder, x, y, yaw, color, label):
    hx, hy, hz = BIN_HALF
    vol = 8 * hx * hy * hz
    body = builder.add_link(xform=wp.transform(p=wp.vec3(x, y, hz + 0.002), q=_quat_z(yaw)), label=label)
    # Low effective friction: a real cart rides on its rear wheels with only the front lip sliding.
    cfg = newton.ModelBuilder.ShapeConfig(density=BIN_MASS / vol, mu=0.25, gap=0.01)
    builder.add_shape_box(body, hx=hx, hy=hy, hz=hz, cfg=cfg, color=color, label=label)
    # Lid and the latch bar the hook grabs.
    builder.add_shape_box(
        body, xform=wp.transform(p=wp.vec3(0.03, 0.0, hz + 0.02)), hx=hx + 0.04, hy=hy + 0.01, hz=0.02,
        cfg=_visual(), color=tuple(0.8 * c for c in color), label=f"{label}_lid",
    )
    builder.add_shape_capsule(
        body, xform=wp.transform(p=wp.vec3(*LATCH_LOCAL), q=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * math.pi)),
        radius=0.015, half_height=hy * 0.6, cfg=_visual(), color=(0.15, 0.15, 0.15), label=f"{label}_latch",
    )
    joint = builder.add_joint_free(body, label=f"{label}_free")
    builder.add_articulation([joint], label=label)
    return body, joint


def _segments_box(builder, pts, width, z, color, label):
    for i, (a, b) in enumerate(zip(pts, pts[1:])):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        cx, cy = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        # Overlap segments a bit so bends have no gaps.
        builder.add_shape_box(
            -1, xform=wp.transform(p=wp.vec3(cx, cy, z), q=_quat_z(math.atan2(dy, dx))),
            hx=length / 2 + width * 0.3, hy=width / 2, hz=0.002, cfg=_visual(), color=color, label=f"{label}_{i}",
        )


def _dist_to_polyline(p, pts):
    best = math.inf
    for a, b in zip(pts, pts[1:]):
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy or 1.0)))
        best = min(best, math.hypot(p[0] - ax - t * dx, p[1] - ay - t * dy))
    return best


def build_flat(builder, rng):
    builder.add_ground_plane(color=GRASS, cfg=GROUND_CFG)
    _segments_box(builder, [(-3.0, 0.0), (12.0, 0.0)], 3.0, 0.004, ASPHALT, "pad")
    return SceneSpec(
        name="flat",
        start=(0.0, 0.0, 0.0),
        bins=[(8.0, 0.6, math.pi, (0.12, 0.30, 0.16)), (8.0, -0.6, math.pi, (0.12, 0.25, 0.55))],
        route=[(0.0, 0.0), (6.5, 0.6)],
        bounds=(-4.0, -6.0, 12.0, 6.0),
    )


def build_anna_pl(builder, rng):
    builder.add_ground_plane(color=GRASS, cfg=GROUND_CFG)
    paved = []
    for i, road in enumerate(site.PAVEMENT):
        pts = [site.px_to_m(p) for p in road["px"]]
        paved.append((pts, road["width"]))
        _segments_box(builder, pts, road["width"], 0.004 + 0.001 * i, ASPHALT, f"road{i}")

    for i, h in enumerate(site.HOUSES):
        cx, cy = site.px_to_m(h["center"])
        w, d = (s * site.M_PER_PX for s in h["size"])
        builder.add_shape_box(
            -1, xform=wp.transform(p=wp.vec3(cx, cy, h["height"] / 2), q=_quat_z(math.radians(-h["rot"]))),
            hx=w / 2, hy=d / 2, hz=h["height"] / 2, color=(0.78, 0.76, 0.72), label=f"house{i}",
        )

    trunk_cfg = newton.ModelBuilder.ShapeConfig(mu=0.8, gap=0.01)
    n = 0
    for (cpx, r_px) in site.TREE_CLUSTERS:
        c = site.px_to_m(cpx)
        r = r_px * site.M_PER_PX
        for _ in range(int(3 + r * 1.2)):
            a, d = rng.uniform(0, 2 * math.pi), r * math.sqrt(rng.random())
            p = (c[0] + d * math.cos(a), c[1] + d * math.sin(a))
            if any(_dist_to_polyline(p, pts) < w / 2 + 1.5 for pts, w in paved):
                continue
            height = rng.uniform(6.0, 14.0)
            crown = rng.uniform(2.0, 3.5)
            builder.add_shape_cylinder(
                -1, xform=wp.transform(p=wp.vec3(p[0], p[1], height / 2)), radius=rng.uniform(0.12, 0.3),
                half_height=height / 2, cfg=trunk_cfg, color=(0.30, 0.22, 0.14), label=f"trunk{n}",
            )
            builder.add_shape_sphere(
                -1, xform=wp.transform(p=wp.vec3(p[0], p[1], height)), radius=crown, cfg=_visual(),
                color=(0.13 + rng.uniform(-0.03, 0.03), 0.30 + rng.uniform(-0.05, 0.05), 0.12), label=f"crown{n}",
            )
            n += 1

    route = site.route_m()
    bins = []
    for (bx, by), color in zip(site.bins_m(), [(0.12, 0.30, 0.16), (0.12, 0.25, 0.55)]):
        bins.append((bx, by, math.radians(site.BIN_YAW_DEG), color))
    xs = [p[0] for p in route]
    ys = [p[1] for p in route]
    return SceneSpec(
        name="anna_pl",
        start=site.start_pose(),
        bins=bins,
        route=route,
        bounds=(min(xs) - 15, min(ys) - 10, max(xs) + 15, max(ys) + 10),
    )


SCENES = {"flat": build_flat, "anna_pl": build_anna_pl}


# ---------------------------------------------------------------------------
# Kernels


@wp.kernel
def hitch_force(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_f: wp.array[wp.spatial_vector],
    hitch: wp.array[wp.int32],  # [car_body, bin_body or -1]
    hook_local: wp.vec3,
    latch_local: wp.vec3,
    k: float,
    c: float,
):
    a = hitch[0]
    b = hitch[1]
    if b < 0:
        return
    qa = body_q[a]
    qb = body_q[b]
    pa = wp.transform_point(qa, hook_local)
    pb = wp.transform_point(qb, latch_local)
    ca = wp.transform_point(qa, body_com[a])
    cb = wp.transform_point(qb, body_com[b])
    va = wp.spatial_top(body_qd[a]) + wp.cross(wp.spatial_bottom(body_qd[a]), pa - ca)
    vb = wp.spatial_top(body_qd[b]) + wp.cross(wp.spatial_bottom(body_qd[b]), pb - cb)
    f = k * (pa - pb) + c * (va - vb)  # force on the bin, toward the hook
    wp.atomic_add(body_f, b, wp.spatial_vector(f, wp.cross(pb - cb, f)))
    wp.atomic_add(body_f, a, wp.spatial_vector(-f, wp.cross(pa - ca, -f)))


# ---------------------------------------------------------------------------


class Sim:
    def __init__(self, scene="flat", seed=0):
        self.scene_name = scene
        self.seed = seed
        self._build(scene, seed)

    # -- construction -------------------------------------------------------

    def _build(self, scene, seed):
        rng = random.Random(seed)
        builder = newton.ModelBuilder()
        builder.default_joint_cfg.damping = 0.01
        self.spec = SCENES[scene](builder, rng)
        self.ground_shapes = [0]

        sx, sy, syaw = self.spec.start
        self.car, self.car_joint, self.drive_joints = add_car(builder, sx, sy, syaw)
        self.bins = []
        for i, (bx, by, byaw, color) in enumerate(self.spec.bins):
            body, joint = add_bin(builder, bx, by, byaw, color, f"bin{i}")
            self.bins.append({"body": body, "joint": joint})

        self.model = builder.finalize()
        self.device = self.model.device
        self.solver = newton.solvers.SolverMuJoCo(
            self.model, use_mujoco_contacts=False, disable_sensors=True, solver="newton",
            integrator="implicitfast", cone="elliptic", iterations=20, ls_iterations=50,
            njmax=2048, nconmax=1024,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision = newton.CollisionPipeline(self.model, rigid_contact_max=2048)
        self.contacts = self.collision.contacts()

        q_start = self.model.joint_q_start.numpy()
        qd_start = self.model.joint_qd_start.numpy()
        self.drive_dofs = [int(qd_start[j]) for j in self.drive_joints]
        self.car_q0 = int(q_start[self.car_joint])
        for b in self.bins:
            b["q0"] = int(q_start[b["joint"]])
        self.joint_q_init = self.model.joint_q.numpy().copy()

        self.hitch = wp.array([self.car, -1], dtype=wp.int32, device=self.device)
        self._target_qd = np.zeros(self.model.joint_dof_count, dtype=np.float32)

        self._init_cameras()
        self.reset(seed=seed, randomize=False)
        self._capture()

    def _init_cameras(self):
        self.camera = SensorTiledCamera(model=self.model)
        cfg = self.camera.default_render_config
        cfg.enable_shadows = True
        self.camera.utils.create_default_light(enable_shadows=True)
        self.clear = SensorTiledCamera.ClearData(clear_color=SKY, clear_albedo=SKY)
        self.cam_buffers = {}
        for name, (w, h, fov) in CAMERAS.items():
            u = self.camera.utils
            self.cam_buffers[name] = {
                "rays": u.compute_camera_rays_pinhole(w, h, camera_fovs=math.radians(fov)),
                "color": u.create_color_image_output(w, h, 1),
                "depth": u.create_depth_image_output(w, h, 1),
            }

    def _capture(self):
        self.graph = None
        if self.device.is_cuda:
            with wp.ScopedCapture() as cap:
                self._simulate()
            self.graph = cap.graph

    # -- stepping -----------------------------------------------------------

    def _simulate(self):
        dt = 1.0 / (FPS * SUBSTEPS)
        for _ in range(SUBSTEPS):
            self.state_0.clear_forces()
            wp.launch(
                hitch_force, dim=1,
                inputs=[self.state_0.body_q, self.state_0.body_qd, self.model.body_com, self.state_0.body_f,
                        self.hitch, wp.vec3(*HOOK_LOCAL), wp.vec3(*LATCH_LOCAL), HITCH_K, HITCH_C],
            )
            self.collision.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self, frames=1):
        for _ in range(frames):
            if self.cmd_timeout > 0 and self.time - self.cmd_time > self.cmd_timeout and self.cmd != (0.0, 0.0):
                self.set_drive(0.0, 0.0, _from_watchdog=True)
            if self.graph is not None:
                wp.capture_launch(self.graph)
            else:
                self._simulate()
            self.time += 1.0 / FPS
            self.frame += 1

    def reset(self, seed=None, randomize=True):
        if seed is not None:
            self.seed = seed
        rng = random.Random(self.seed)
        q = self.joint_q_init.copy()

        def place(q0, x, y, z, yaw):
            q[q0:q0 + 3] = (x, y, z)
            h = 0.5 * yaw
            q[q0 + 3:q0 + 7] = (0.0, 0.0, math.sin(h), math.cos(h))

        sx, sy, syaw = self.spec.start
        if randomize:
            sx += rng.uniform(-0.5, 0.5)
            sy += rng.uniform(-0.5, 0.5)
            syaw += math.radians(rng.uniform(-15, 15))
        place(self.car_q0, sx, sy, CHASSIS_Z0, syaw)
        for b, (bx, by, byaw, _c) in zip(self.bins, self.spec.bins):
            if randomize:
                # Bins sit side by side, so keep the jitter small enough that they never overlap.
                bx += rng.uniform(-0.25, 0.25)
                by += rng.uniform(-0.15, 0.15)
                byaw += math.radians(rng.uniform(-20, 20))
            place(b["q0"], bx, by, BIN_HALF[2] + 0.002, byaw)

        qd = np.zeros(self.model.joint_dof_count, dtype=np.float32)
        for st in (self.state_0, self.state_1):
            st.joint_q.assign(q)
            st.joint_qd.assign(qd)
            newton.eval_fk(self.model, st.joint_q, st.joint_qd, st)
        self.hitch.assign(np.array([self.car, -1], dtype=np.int32))
        self.latched_bin = None
        self.time = 0.0
        self.frame = 0
        self.cmd = (0.0, 0.0)
        self.cmd_time = 0.0
        self.cmd_timeout = getattr(self, "cmd_timeout", 1.0)
        self.set_drive(0.0, 0.0)

    # -- actuation ----------------------------------------------------------

    def set_drive(self, v, w, _from_watchdog=False):
        """Body-frame command: v forward (m/s), w yaw rate (rad/s, + is left)."""
        left = v - w * TRACK / 2
        right = v + w * TRACK / 2
        scale = max(1.0, abs(left) / MAX_WHEEL_SPEED, abs(right) / MAX_WHEEL_SPEED)
        left, right = left / scale, right / scale
        # Positive rotation about the +Y axle rolls the wheel toward +X.
        self._target_qd[self.drive_dofs[0]] = left / WHEEL_RADIUS
        self._target_qd[self.drive_dofs[1]] = right / WHEEL_RADIUS
        self.control.joint_target_qd.assign(self._target_qd)
        self.cmd = (float(v), float(w))
        if not _from_watchdog:
            self.cmd_time = self.time
        return {"left": left, "right": right}

    def latch(self, engage=True):
        if not engage:
            self.hitch.assign(np.array([self.car, -1], dtype=np.int32))
            self.latched_bin = None
            return {"latched": None}
        best = self.nearest_latch()
        if best["distance"] <= LATCH_RANGE:
            self.hitch.assign(np.array([self.car, self.bins[best["bin"]]["body"]], dtype=np.int32))
            self.latched_bin = best["bin"]
        return {"latched": self.latched_bin, "nearest": best}

    def nearest_latch(self):
        bq = self.state_0.body_q.numpy()
        hook = self._world_point(bq[self.car], HOOK_LOCAL)
        best = {"bin": None, "distance": math.inf}
        for i, b in enumerate(self.bins):
            d = float(np.linalg.norm(hook - self._world_point(bq[b["body"]], LATCH_LOCAL)))
            if d < best["distance"]:
                best = {"bin": i, "distance": d}
        return best

    @staticmethod
    def _world_point(tf, local):
        return np.asarray(tf[:3]) + _rotate(tf[3:7], local)

    # -- observation --------------------------------------------------------

    def get_state(self):
        bq = self.state_0.body_q.numpy()
        bqd = self.state_0.body_qd.numpy()
        car = bq[self.car]
        fwd = _rotate(car[3:7], (1.0, 0.0, 0.0))
        out = {
            "scene": self.scene_name,
            "seed": self.seed,
            "time": round(self.time, 4),
            "frame": self.frame,
            "car": {
                "x": float(car[0]), "y": float(car[1]), "z": float(car[2]),
                "yaw": _yaw_of(car[3:7]),
                "v": float(np.dot(bqd[self.car][:3], fwd)),
                "w": float(bqd[self.car][5]),
                "upright": float(_rotate(car[3:7], (0.0, 0.0, 1.0))[2]),
                "hook": self._world_point(car, HOOK_LOCAL).tolist(),
                "cmd": {"v": self.cmd[0], "w": self.cmd[1]},
            },
            "bins": [],
            "latched": self.latched_bin,
        }
        for i, b in enumerate(self.bins):
            t = bq[b["body"]]
            out["bins"].append({
                "id": i, "x": float(t[0]), "y": float(t[1]), "yaw": _yaw_of(t[3:7]),
                "upright": float(_rotate(t[3:7], (0.0, 0.0, 1.0))[2]),
                "latch": self._world_point(t, LATCH_LOCAL).tolist(),
            })
        return out

    def info(self):
        return {
            "scene": self.scene_name,
            "scenes": list(SCENES),
            "cameras": {k: {"width": w, "height": h, "fov": f} for k, (w, h, f) in CAMERAS.items()},
            # Body-frame mount of the front camera (OpenGL convention: looks down -Z, +Y up),
            # plus the chassis-center height, so clients can turn depth into world points.
            "front_mount": {"pos": FRONT_CAM_POS, "dir": FRONT_CAM_DIR, "chassis_z": CHASSIS_Z0},
            "route": self.spec.route,
            "bounds": self.spec.bounds,
            "fps": FPS,
            "car": {"wheel_radius": WHEEL_RADIUS, "track": TRACK, "max_wheel_speed": MAX_WHEEL_SPEED,
                    "hook_local": HOOK_LOCAL},
            "bin": {"half_extents": BIN_HALF, "latch_local": LATCH_LOCAL, "latch_range": LATCH_RANGE},
        }

    def _camera_pose(self, name, car):
        p = np.asarray(car[:3])
        q = car[3:7]
        if name == "front":
            eye = p + _rotate(q, FRONT_CAM_POS)
            target = eye + _rotate(q, FRONT_CAM_DIR)
        elif name == "chase":
            yaw = _yaw_of(q)
            eye = p + np.array([-3.0 * math.cos(yaw), -3.0 * math.sin(yaw), 1.8])
            target = p + np.array([1.5 * math.cos(yaw), 1.5 * math.sin(yaw), 0.0])
        else:  # overhead: fit the scene bounds
            x0, y0, x1, y1 = self.spec.bounds
            span = max(x1 - x0, y1 - y0)
            height = span / 2 / math.tan(math.radians(CAMERAS["overhead"][2]) / 2) + 2
            eye = np.array([(x0 + x1) / 2, (y0 + y1) / 2, height])
            target = eye - np.array([0.0, 0.0, 1.0])
        return eye, look_at_quat(eye, target)

    def render(self, name):
        """Returns (rgb uint8 HxWx3, depth float32 HxW meters, -1 = no hit)."""
        buf = self.cam_buffers[name]
        car = self.state_0.body_q.numpy()[self.car]
        eye, quat = self._camera_pose(name, car)
        tf = wp.array([[wp.transformf(wp.vec3f(*eye), wp.quatf(*quat))]], dtype=wp.transformf, device=self.device)
        self.model.bvh_refit_shapes(self.state_0)
        self.camera.update(
            self.state_0, tf, buf["rays"], color_image=buf["color"], depth_image=buf["depth"], clear_data=self.clear,
        )
        w, h, _ = CAMERAS[name]
        rgba = buf["color"].numpy().view(np.uint8).reshape(h, w, 4)
        return rgba[..., :3].copy(), buf["depth"].numpy().reshape(h, w).copy()

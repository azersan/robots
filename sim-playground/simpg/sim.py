"""Newton-backed simulator for the driveway car.

World frame: Z up, X east, Y north, meters. The car's body frame is X forward,
Y left, Z up. Everything here runs on one thread at a time; the gateway holds
a lock around every call.
"""

import copy
import math
import random
from dataclasses import dataclass, field

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorTiledCamera

from . import meshes
from . import site_anna_pl as site
from .terrain import Terrain

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

# Wheelie carts. Body frame: +X is the back (vertical face with the wheels and handle, the side
# that faces the house at the curb), so that's where the robot hooks on.
BIN_TYPES = {
    # 96 gal: ~34" deep, 26.5" wide, 43" tub + lid
    "recycling": {"depth": 0.86, "width": 0.67, "height": 1.10, "mass": 15.0, "color": (0.10, 0.27, 0.62)},
    # 64 gal: ~29" deep, 24" wide, 40" tub + lid
    "trash": {"depth": 0.74, "width": 0.61, "height": 1.00, "mass": 12.0, "color": (0.40, 0.42, 0.44)},
}
LATCH_OUT = 0.06  # latch bar sits this far behind the back face (between the wheels)
LATCH_Z = 0.08  # and this high off the ground
LATCH_RANGE = 0.08  # hook must be this close to a latch bar to engage
CART_WHEEL_R = 0.10


def bin_half(kind):
    t = BIN_TYPES[kind]
    return (t["depth"] / 2, t["width"] / 2, t["height"] / 2)


def latch_local(kind):
    hx, _, hz = bin_half(kind)
    return (hx + LATCH_OUT, 0.0, LATCH_Z - hz)

HITCH_K = 4000.0
HITCH_C = 150.0

FPS = 60
SUBSTEPS = 10
RECORD_EVERY = 6  # episode log: one ground-truth row every 0.1 s
COMPASS_NOISE_DEG = 2.0

# Contact friction is the max of the two surfaces, so the ground stays low and each
# object's own mu (tires 1.0, bin 0.25, ...) decides.
GROUND_CFG = newton.ModelBuilder.ShapeConfig(mu=0.2, gap=0.01)

SKY = 0xFFEBCDA8  # packed 0xAABBGGRR: a hazy afternoon blue
SUN_DIR = (0.48, 0.55, -0.68)  # direction the light travels: sun in the south-west, ~43 deg up
SUPERSAMPLE = {"front": 2, "rear": 2, "chase": 2, "overhead": 1}  # renders at N x N and downsamples (no texture mipmaps)

# Fallback flat colors when the photo textures haven't been fetched (tools/fetch_assets.py).
FALLBACK = {"grass": (0.33, 0.47, 0.22), "concrete": (0.70, 0.68, 0.64), "asphalt": (0.36, 0.36, 0.38),
            "bark": (0.30, 0.22, 0.14)}

# Front camera mount in the chassis frame: position and look direction (pitched down ~7 deg).
FRONT_CAM_POS = (CHASSIS_HALF[0] + 0.01, 0.0, CHASSIS_HALF[2] + 0.06)
FRONT_CAM_DIR = (1.0, 0.0, -0.12)
# Rear camera: the mirror image, on the tail, looking backward. It sees where you're going when
# reversing with a bin on the nose.
REAR_CAM_POS = (-(CHASSIS_HALF[0] + 0.01), 0.0, CHASSIS_HALF[2] + 0.06)
REAR_CAM_DIR = (-1.0, 0.0, -0.12)
ROBOT_CAMERAS = {"front": (FRONT_CAM_POS, FRONT_CAM_DIR), "rear": (REAR_CAM_POS, REAR_CAM_DIR)}

CAMERAS = {
    # name: (width, height, vertical fov deg)
    "front": (320, 240, 70.0),
    "rear": (320, 240, 70.0),
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
    directions: str = ""  # plain-language directions for a robot that only has its camera
    terrain: Terrain | None = None  # scanned ground; None = flat at z = 0


# Footprints used to spawn things on sloped ground without starting inside it (half x, half y).
CAR_FOOTPRINT = (CHASSIS_HALF[0], TRACK / 2 + WHEEL_HALF_WIDTH)


def ground_under(terrain, x, y, yaw=0.0, half=(0.0, 0.0)):
    """Ground height for placing something at (x, y): 0 when flat, else the highest point under its footprint."""
    return 0.0 if terrain is None else terrain.support(x, y, yaw, *half)


def add_car(builder, x, y, yaw, z0=0.0):
    chassis = builder.add_link(
        xform=wp.transform(p=wp.vec3(x, y, z0 + CHASSIS_Z0), q=_quat_z(yaw)), label="car"
    )
    # ~17 kg chassis: a real tug carries its battery low for traction.
    body_cfg = newton.ModelBuilder.ShapeConfig(density=600.0, mu=0.6, gap=0.01)
    hx, hy, hz = CHASSIS_HALF
    builder.add_shape_box(chassis, hx=hx, hy=hy, hz=hz, cfg=body_cfg, color=(0.85, 0.45, 0.10), label="car_body")
    # Camera housings (front and rear) and tow hook, for looks.
    for sx in (1.0, -1.0):
        builder.add_shape_box(
            chassis, xform=wp.transform(p=wp.vec3(sx * (hx - 0.04), 0.0, hz + 0.04)), hx=0.03, hy=0.05, hz=0.04,
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


_ROT_X90 = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * math.pi)


class Look:
    """Textures shared across a scene (loaded once; None means "use a flat color")."""

    def __init__(self):
        rgba = meshes.rgba
        self.photo = {k: rgba(meshes.texture(k)) for k in FALLBACK}
        if self.photo["grass"] is not None:
            self.photo["grass"] = meshes.grade(self.photo["grass"], saturation=0.7, gain=(1.0, 0.92, 0.75))
        self.leaves = rgba(meshes.leaves_texture())
        self.plastic = rgba(meshes.plastic_texture())
        self.siding = rgba(meshes.siding_texture())
        self.shingles = rgba(meshes.shingles_texture())

    def surface(self, kind):
        """(texture, color) for a ground surface: the photo as-is, or a flat fallback color."""
        tex = self.photo[kind]
        return (tex, (1.0, 1.0, 1.0)) if tex is not None else (None, FALLBACK[kind])


def add_bin(builder, x, y, yaw, kind, label, look, z0=0.0):
    t = BIN_TYPES[kind]
    hx, hy, hz = bin_half(kind)
    body = builder.add_link(xform=wp.transform(p=wp.vec3(x, y, z0 + hz + 0.002), q=_quat_z(yaw)), label=label)
    # Physics: one box. Low effective friction: a real cart rides on its wheels with only the lip sliding.
    cfg = newton.ModelBuilder.ShapeConfig(density=t["mass"] / (8 * hx * hy * hz), mu=0.25, gap=0.01)
    hidden = copy.copy(cfg)
    hidden.is_visible = False
    builder.add_shape_box(body, hx=hx, hy=hy, hz=hz, cfg=hidden, label=label)

    color = t["color"]
    dark = (0.07, 0.07, 0.08)
    # Tub, lid, handle.
    tub = meshes.cart_body(t["depth"], t["width"], t["height"] - 0.04)
    tub.texture = look.plastic
    builder.add_shape_mesh(body, xform=wp.transform(p=wp.vec3(0.0, 0.0, -hz)), mesh=tub, cfg=_visual(),
                           color=color, label=f"{label}_tub")
    lid = meshes.cart_lid(t["depth"], t["width"])
    lid.texture = look.plastic
    builder.add_shape_mesh(body, xform=wp.transform(p=wp.vec3(-0.01, 0.0, hz - 0.04)), mesh=lid, cfg=_visual(),
                           color=tuple(0.85 * c for c in color), label=f"{label}_lid")
    builder.add_shape_capsule(
        body, xform=wp.transform(p=wp.vec3(hx + 0.035, 0.0, hz - 0.07), q=_ROT_X90),
        radius=0.018, half_height=hy * 0.75, cfg=_visual(), color=tuple(0.8 * c for c in color),
        label=f"{label}_handle",
    )
    for s in (-1.0, 1.0):  # handle side arms
        builder.add_shape_box(
            body, xform=wp.transform(p=wp.vec3(hx + 0.018, s * hy * 0.75, hz - 0.11)), hx=0.018, hy=0.02, hz=0.05,
            cfg=_visual(), color=tuple(0.8 * c for c in color), label=f"{label}_handle_arm",
        )
    # Wheels and axle at the back bottom, and the latch bar between them.
    wheel_x = hx - 0.02
    for s in (-1.0, 1.0):
        builder.add_shape_cylinder(
            body, xform=wp.transform(p=wp.vec3(wheel_x, s * (hy - 0.035), CART_WHEEL_R - hz), q=_ROT_X90),
            radius=CART_WHEEL_R, half_height=0.03, cfg=_visual(), color=dark, label=f"{label}_wheel",
        )
        builder.add_shape_cylinder(
            body, xform=wp.transform(p=wp.vec3(wheel_x, s * (hy - 0.005), CART_WHEEL_R - hz), q=_ROT_X90),
            radius=0.035, half_height=0.004, cfg=_visual(), color=(0.55, 0.55, 0.57), label=f"{label}_hub",
        )
    builder.add_shape_capsule(
        body, xform=wp.transform(p=wp.vec3(wheel_x, 0.0, CART_WHEEL_R - hz), q=_ROT_X90),
        radius=0.012, half_height=hy - 0.07, cfg=_visual(), color=(0.45, 0.45, 0.47), label=f"{label}_axle",
    )
    lx, _, lz = latch_local(kind)
    builder.add_shape_capsule(
        body, xform=wp.transform(p=wp.vec3(lx, 0.0, lz), q=_ROT_X90),
        radius=0.012, half_height=0.12, cfg=_visual(), color=(0.75, 0.62, 0.10), label=f"{label}_latch",
    )
    for s in (-1.0, 1.0):  # brackets holding the latch bar off the tub
        builder.add_shape_box(
            body, xform=wp.transform(p=wp.vec3((hx + lx) / 2, s * 0.12, lz)), hx=(lx - hx) / 2 + 0.01, hy=0.01,
            hz=0.012, cfg=_visual(), color=(0.75, 0.62, 0.10), label=f"{label}_latch_bracket",
        )
    joint = builder.add_joint_free(body, label=f"{label}_free")
    builder.add_articulation([joint], label=label)
    return body, joint


def add_ground(builder, look, bounds, margin=250.0, terrain=None):
    """Invisible physics ground plus textured grass well past the horizon trees. With terrain: a heightfield
    collider and grass draped over it; the plane and far grass sit just below its lowest point."""
    plane = copy.copy(GROUND_CFG)
    plane.is_visible = False
    base = 0.0 if terrain is None else float(terrain.z.min()) - 0.05
    builder.add_ground_plane(height=base, cfg=plane)
    x0, y0, x1, y1 = bounds
    tex, color = look.surface("grass")
    builder.add_shape_mesh(
        -1, mesh=meshes.ground_quad(x0 - margin, y0 - margin, x1 + margin, y1 + margin, tile=2.0, z=base, tex=tex),
        cfg=_visual(), color=color, label="grass",
    )
    if terrain is not None:
        terrain.add_physics(builder, plane)
        tx0, ty0, tx1, ty1 = terrain.bounds
        builder.add_shape_mesh(
            # 3 cm under the true surface: the grass sheet and the pavement follow the terrain with different
            # triangles, and on bumpy ground the grass would otherwise poke through the pavement.
            -1, mesh=meshes.ground_grid(tx0, ty0, tx1, ty1, step=0.5, tile=2.0,
                                        height_fn=lambda x, y: terrain.height(x, y) - 0.03, tex=tex),
            cfg=_visual(), color=color, label="grass_terrain",
        )


def add_treeline(builder, look, bounds, rng, inner=35.0, outer=70.0, count=140, terrain=None):
    """A ring of big crowns around the lot so the horizon reads as woods, not the edge of the world."""
    x0, y0, x1, y1 = bounds
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rx, ry = (x1 - x0) / 2, (y1 - y0) / 2
    tex, color = look.surface("bark")
    for i in range(count):
        a = 2 * math.pi * (i + rng.random()) / count
        d = rng.uniform(inner, outer)
        x, y = cx + (rx + d) * math.cos(a), cy + (ry + d) * math.sin(a)
        height = rng.uniform(14.0, 22.0)
        trunk = meshes.tapered_cylinder(0.35, 0.18, height * 0.7, segments=8, tile=(0.8, 1.5), tex=tex)
        gz = ground_under(terrain, x, y)
        builder.add_shape_mesh(-1, xform=wp.transform(p=wp.vec3(x, y, gz)), mesh=trunk, cfg=_visual(),
                               color=tuple(0.8 * c for c in color), label=f"edge{i}_bark")
        crown = meshes.canopy(rng.uniform(5.5, 8.5), np.random.default_rng(rng.randrange(1 << 30)), lobes=6,
                              tex=look.leaves)
        tint = (rng.uniform(0.75, 1.0), rng.uniform(0.8, 1.05), rng.uniform(0.65, 0.95))
        builder.add_shape_mesh(-1, xform=wp.transform(p=wp.vec3(x, y, gz + height * 0.65)), mesh=crown, cfg=_visual(),
                               color=tint, label=f"edge{i}_crown")


def add_pavement(builder, look, pts, width, kind, z, label, tile=3.0, terrain=None):
    tex, color = look.surface(kind)
    # On terrain, ride 1 cm above the surface so the draped grass never shows through.
    mesh = meshes.ribbon(pts, width, tile, z=z if terrain is None else z + 0.01, tex=tex,
                         height_fn=None if terrain is None else terrain.height)
    builder.add_shape_mesh(-1, mesh=mesh, cfg=_visual(), color=color, label=label)


def add_tree(builder, look, x, y, rng, label, terrain=None):
    height = rng.uniform(11.0, 20.0)
    r = rng.uniform(0.16, 0.38)
    gz = ground_under(terrain, x, y)
    # Physics: a plain cylinder. Visual: tapered bark trunk and a clumpy crown up high.
    builder.add_shape_cylinder(
        -1, xform=wp.transform(p=wp.vec3(x, y, gz + 1.5)), radius=r, half_height=1.5,
        cfg=newton.ModelBuilder.ShapeConfig(mu=0.8, gap=0.01, is_visible=False), label=f"{label}_trunk",
    )
    tex, color = look.surface("bark")
    trunk = meshes.tapered_cylinder(r, r * 0.5, height * 0.75, segments=12, tile=(0.6, 1.2), tex=tex,
                                    jitter=0.06, rng=rng)
    builder.add_shape_mesh(-1, xform=wp.transform(p=wp.vec3(x, y, gz)), mesh=trunk, cfg=_visual(),
                           color=tuple(c * rng.uniform(0.75, 1.0) for c in color), label=f"{label}_bark")
    crown = meshes.canopy(rng.uniform(3.5, 5.5), np.random.default_rng(rng.randrange(1 << 30)), lobes=7,
                          tex=look.leaves)
    tint = (rng.uniform(0.8, 1.05), rng.uniform(0.85, 1.1), rng.uniform(0.7, 1.0))
    builder.add_shape_mesh(-1, xform=wp.transform(p=wp.vec3(x, y, gz + height * 0.68)), mesh=crown, cfg=_visual(),
                           color=tint, label=f"{label}_crown")


def add_house(builder, look, cx, cy, length, width, rot, wall_h, label, garage=False, terrain=None):
    """Box house with siding, a gable roof, windows, and optionally two garage doors on the +X end."""
    q = _quat_z(rot)
    gz = 0.0
    if terrain is not None:
        c, s = math.cos(rot), math.sin(rot)
        lx = np.array([1, 1, -1, -1, 0]) * length / 2
        ly = np.array([1, -1, 1, -1, 0]) * width / 2
        gz = float(np.min(terrain.height(cx + c * lx - s * ly, cy + s * lx + c * ly)))
    # Physics: one box.
    builder.add_shape_box(
        -1, xform=wp.transform(p=wp.vec3(cx, cy, gz + wall_h / 2), q=q), hx=length / 2, hy=width / 2, hz=wall_h / 2,
        cfg=newton.ModelBuilder.ShapeConfig(gap=0.01, is_visible=False), label=label,
    )
    walls = meshes.loft([(0.0, meshes._rounded_rect(length / 2, width / 2, 0.01, n=1)),
                         (wall_h, meshes._rounded_rect(length / 2, width / 2, 0.01, n=1))], tile=2.0, cap_bottom=False)
    walls.texture = look.siding
    builder.add_shape_mesh(-1, xform=wp.transform(p=wp.vec3(cx, cy, gz), q=q), mesh=walls, cfg=_visual(),
                           color=(0.86, 0.84, 0.78), label=f"{label}_walls")
    roof = meshes.gable_roof(length, width, rise=width * 0.3)
    roof.texture = look.shingles
    builder.add_shape_mesh(-1, xform=wp.transform(p=wp.vec3(cx, cy, gz + wall_h), q=q), mesh=roof, cfg=_visual(),
                           color=(0.30, 0.29, 0.30), label=f"{label}_roof")

    def on_wall(lx, ly, lz):
        c, s = math.cos(rot), math.sin(rot)
        return wp.vec3(cx + c * lx - s * ly, cy + s * lx + c * ly, gz + lz)

    win = (0.12, 0.15, 0.18)
    trim = (0.95, 0.95, 0.93)
    for side in (-1.0, 1.0):
        for k in range(-2, 3):
            for z in (1.4, wall_h - 1.6):
                lx = k * length / 5.5
                p = on_wall(lx, side * (width / 2 + 0.02), z)
                builder.add_shape_box(-1, xform=wp.transform(p=p, q=q), hx=0.55, hy=0.03, hz=0.7, cfg=_visual(),
                                      color=trim, label=f"{label}_trim")
                p = on_wall(lx, side * (width / 2 + 0.05), z)
                builder.add_shape_box(-1, xform=wp.transform(p=p, q=q), hx=0.45, hy=0.02, hz=0.6, cfg=_visual(),
                                      color=win, label=f"{label}_window")
    if garage:
        for k in (-1.0, 1.0):
            p = on_wall(length / 2 + 0.03, k * min(1.6, width / 4), 1.1)
            builder.add_shape_box(-1, xform=wp.transform(p=p, q=q), hx=0.03, hy=1.25, hz=1.1, cfg=_visual(),
                                  color=(0.92, 0.92, 0.90), label=f"{label}_garage")
            for j in range(4):  # door panel seams
                p = on_wall(length / 2 + 0.065, k * min(1.6, width / 4), 0.3 + j * 0.55)
                builder.add_shape_box(-1, xform=wp.transform(p=p, q=q), hx=0.005, hy=1.2, hz=0.012, cfg=_visual(),
                                      color=(0.7, 0.7, 0.68), label=f"{label}_garage_seam")


def _dist_to_polyline(p, pts):
    best = math.inf
    for a, b in zip(pts, pts[1:]):
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy or 1.0)))
        best = min(best, math.hypot(p[0] - ax - t * dx, p[1] - ay - t * dy))
    return best


def build_flat(builder, rng, look):
    bounds = (-4.0, -6.0, 12.0, 6.0)
    add_ground(builder, look, bounds)
    add_pavement(builder, look, [(-3.0, 0.0), (12.0, 0.0)], 3.2, "concrete", 0.004, "pad", tile=2.5)
    return SceneSpec(
        name="flat",
        start=(0.0, 0.0, 0.0),
        bins=[(8.0, 0.65, math.pi, "recycling"), (8.0, -0.6, math.pi, "trash")],
        route=[(0.0, 0.0), (6.5, 0.6)],
        directions=("You start at one end of a straight concrete pad, facing along it. The two bins, a big blue "
                    "recycling cart and a smaller gray trash cart, stand side by side at the far end of the pad, "
                    "about 8 m ahead, with their backs (handles and wheels) facing you."),
        bounds=bounds,
    )


def build_anna_pl(builder, rng, look, terrain=None, name="anna_pl"):
    route = site.route_m()
    xs = [p[0] for p in route]
    ys = [p[1] for p in route]
    bounds = (min(xs) - 15, min(ys) - 10, max(xs) + 15, max(ys) + 10)
    add_ground(builder, look, bounds, terrain=terrain)

    paved = []
    for i, road in enumerate(site.PAVEMENT):
        pts = [site.px_to_m(p) for p in road["px"]]
        paved.append((pts, road["width"]))
        add_pavement(builder, look, pts, road["width"], road["surface"], 0.004 + 0.001 * i, f"road{i}",
                     tile=2.5 if road["surface"] == "concrete" else 4.0, terrain=terrain)

    for i, h in enumerate(site.HOUSES):
        cx, cy = site.px_to_m(h["center"])
        w, d = (s * site.M_PER_PX for s in h["size"])
        add_house(builder, look, cx, cy, d, w, math.radians(90 - h["rot"]), h["height"] - 2.5, f"house{i}",
                  garage=h.get("garage", False), terrain=terrain)

    n = 0
    for (cpx, r_px) in site.TREE_CLUSTERS:
        c = site.px_to_m(cpx)
        r = r_px * site.M_PER_PX
        for _ in range(int(3 + r * 1.2)):
            a, d = rng.uniform(0, 2 * math.pi), r * math.sqrt(rng.random())
            p = (c[0] + d * math.cos(a), c[1] + d * math.sin(a))
            if any(_dist_to_polyline(p, pts) < w / 2 + 1.5 for pts, w in paved):
                continue
            add_tree(builder, look, p[0], p[1], rng, f"tree{n}", terrain=terrain)
            n += 1

    add_treeline(builder, look, bounds, rng, terrain=terrain)
    bins = [(bx, by, math.radians(site.BIN_YAW_DEG), kind) for (bx, by), kind in zip(site.bins_m(), site.BIN_KINDS)]
    return SceneSpec(name=name, start=site.start_pose(), bins=bins, route=route, bounds=bounds,
                     directions=site.DIRECTIONS, terrain=terrain)


def build_anna_pl_scan(builder, rng, look):
    """Anna Pl on the ground from the phone LiDAR scan (terrain/anna_pl.npz, made by tools/import_scan.py)."""
    return build_anna_pl(builder, rng, look, terrain=Terrain.load("anna_pl"), name="anna_pl_scan")


SCENES = {"flat": build_flat, "anna_pl": build_anna_pl, "anna_pl_scan": build_anna_pl_scan}


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
    latch_local: wp.array[wp.vec3],  # [latch point in the latched bin's frame]
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
    pb = wp.transform_point(qb, latch_local[0])
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
        look = Look()
        self.spec = SCENES[scene](builder, rng, look)

        sx, sy, syaw = self.spec.start
        terrain = self.spec.terrain
        self.car, self.car_joint, self.drive_joints = add_car(
            builder, sx, sy, syaw, z0=ground_under(terrain, sx, sy, syaw, CAR_FOOTPRINT))
        self.bins = []
        for i, (bx, by, byaw, kind) in enumerate(self.spec.bins):
            body, joint = add_bin(builder, bx, by, byaw, kind, f"bin{i}", look,
                                  z0=ground_under(terrain, bx, by, byaw, bin_half(kind)[:2]))
            self.bins.append({"body": body, "joint": joint, "kind": kind, "latch": latch_local(kind)})

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
        self.hitch_latch = wp.zeros(1, dtype=wp.vec3, device=self.device)
        self._target_qd = np.zeros(self.model.joint_dof_count, dtype=np.float32)

        self._init_cameras()
        self.reset(seed=seed, randomize=False)
        self._capture()

    def _init_cameras(self):
        self.camera = SensorTiledCamera(model=self.model)
        cfg = self.camera.default_render_config
        cfg.enable_shadows = True
        cfg.enable_textures = True
        cfg.enable_backface_culling = False  # procedural meshes don't guarantee winding
        d = np.asarray(SUN_DIR, np.float32)
        self.camera.utils.create_default_light(enable_shadows=True, direction=wp.vec3f(*(d / np.linalg.norm(d))))
        self.clear = SensorTiledCamera.ClearData(clear_color=SKY, clear_albedo=SKY)
        self.cam_buffers = {}
        u = self.camera.utils
        for name, (w, h, fov) in CAMERAS.items():
            ss = SUPERSAMPLE.get(name, 1)
            self.cam_buffers[name] = {
                "ss": ss,
                "rays": u.compute_camera_rays_pinhole(w * ss, h * ss, camera_fovs=math.radians(fov)),
                "color": u.create_color_image_output(w * ss, h * ss, 1),
                "depth": u.create_depth_image_output(w * ss, h * ss, 1),
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
                        self.hitch, wp.vec3(*HOOK_LOCAL), self.hitch_latch, HITCH_K, HITCH_C],
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
            if self.frame % RECORD_EVERY == 0:
                self._record()

    # -- episode recording (ground truth, for scoring; never exposed to the robot API) --

    def _record(self):
        bq = self.state_0.body_q.numpy()
        car = bq[self.car]
        row = [round(self.time, 3), float(car[0]), float(car[1]), _yaw_of(car[3:7]),
               -1 if self.latched_bin is None else self.latched_bin]
        for b in self.bins:
            t = bq[b["body"]]
            row += [float(t[0]), float(t[1]), float(_rotate(t[3:7], (0.0, 0.0, 1.0))[2])]
        self.episode.append(row)

    def get_episode(self):
        cols = ["t", "x", "y", "yaw", "latched"]
        for i in range(len(self.bins)):
            cols += [f"bin{i}_x", f"bin{i}_y", f"bin{i}_upright"]
        return {"columns": cols, "rows": self.episode, "start": list(self.start_pose),
                "seed": self.seed, "scene": self.scene_name, "every_s": RECORD_EVERY / FPS}

    def place_car(self, x, y, yaw):
        """Teleport the car (at rest) for tools and tests. Not exposed over HTTP."""
        q = self.state_0.joint_q.numpy()
        q0 = self.car_q0
        q[q0:q0 + 3] = (x, y, ground_under(self.spec.terrain, x, y, yaw, CAR_FOOTPRINT) + CHASSIS_Z0)
        q[q0 + 3:q0 + 7] = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
        qd = np.zeros(self.model.joint_dof_count, dtype=np.float32)
        for st in (self.state_0, self.state_1):
            st.joint_q.assign(q)
            st.joint_qd.assign(qd)
            newton.eval_fk(self.model, st.joint_q, st.joint_qd, st)

    # -- robot-mode sensors (only what the real robot could know) --

    def compass_deg(self):
        """Heading in compass degrees (0 = north, 90 = east), with a little magnetometer noise."""
        yaw = _yaw_of(self.state_0.body_q.numpy()[self.car][3:7])
        heading = (90.0 - math.degrees(yaw)) % 360.0
        return round((heading + self._compass_rng.gauss(0.0, COMPASS_NOISE_DEG)) % 360.0, 1)

    def robot_sensors(self):
        return {"time": round(self.time, 3), "compass_deg": self.compass_deg(),
                "latched": self.latched_bin is not None, "cmd": {"v": self.cmd[0], "w": self.cmd[1]}}

    def robot_latch(self, engage=True):
        """Latch with only a yes/no answer, like a limit switch on the hook."""
        r = self.latch(engage)
        return {"latched": r["latched"] is not None}

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
        terrain = self.spec.terrain
        place(self.car_q0, sx, sy, ground_under(terrain, sx, sy, syaw, CAR_FOOTPRINT) + CHASSIS_Z0, syaw)
        self.start_pose = (sx, sy, syaw)
        self.episode = []
        self._compass_rng = random.Random(self.seed * 7919 + 17)
        for b, (bx, by, byaw, kind) in zip(self.bins, self.spec.bins):
            if randomize:
                # Bins sit side by side, so keep the jitter small enough that they never overlap.
                bx += rng.uniform(-0.25, 0.25)
                by += rng.uniform(-0.15, 0.15)
                byaw += math.radians(rng.uniform(-20, 20))
            place(b["q0"], bx, by, ground_under(terrain, bx, by, byaw, bin_half(kind)[:2]) + bin_half(kind)[2] + 0.002,
                  byaw)

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
            b = self.bins[best["bin"]]
            self.hitch_latch.assign(np.array([b["latch"]], dtype=np.float32))
            self.hitch.assign(np.array([self.car, b["body"]], dtype=np.int32))
            self.latched_bin = best["bin"]
        return {"latched": self.latched_bin, "nearest": best}

    def nearest_latch(self):
        bq = self.state_0.body_q.numpy()
        hook = self._world_point(bq[self.car], HOOK_LOCAL)
        best = {"bin": None, "distance": math.inf}
        for i, b in enumerate(self.bins):
            d = float(np.linalg.norm(hook - self._world_point(bq[b["body"]], b["latch"])))
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
                "id": i, "kind": b["kind"], "x": float(t[0]), "y": float(t[1]), "yaw": _yaw_of(t[3:7]),
                "upright": float(_rotate(t[3:7], (0.0, 0.0, 1.0))[2]),
                "latch": self._world_point(t, b["latch"]).tolist(),
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
            "rear_mount": {"pos": REAR_CAM_POS, "dir": REAR_CAM_DIR, "chassis_z": CHASSIS_Z0},
            "route": self.spec.route,
            "bounds": self.spec.bounds,
            "fps": FPS,
            "car": {"wheel_radius": WHEEL_RADIUS, "track": TRACK, "max_wheel_speed": MAX_WHEEL_SPEED,
                    "hook_local": HOOK_LOCAL},
            # Bin yaw points out of the back face (wheels/handle side), where the latch bar is.
            "bin": {"types": {k: {"depth": t["depth"], "width": t["width"], "height": t["height"]}
                              for k, t in BIN_TYPES.items()},
                    "latch_out": LATCH_OUT, "latch_z": LATCH_Z, "latch_range": LATCH_RANGE},
        }

    def _camera_pose(self, name, car):
        p = np.asarray(car[:3])
        q = car[3:7]
        if name in ROBOT_CAMERAS:
            pos, direction = ROBOT_CAMERAS[name]
            eye = p + _rotate(q, pos)
            target = eye + _rotate(q, direction)
        elif name == "chase":
            yaw = _yaw_of(q)
            eye = p + np.array([-3.0 * math.cos(yaw), -3.0 * math.sin(yaw), 1.8])
            target = p + np.array([1.5 * math.cos(yaw), 1.5 * math.sin(yaw), 0.0])
        else:  # overhead: fit the scene bounds
            x0, y0, x1, y1 = self.spec.bounds
            span = max(x1 - x0, y1 - y0)
            height = span / 2 / math.tan(math.radians(CAMERAS["overhead"][2]) / 2) + 2
            top = 0.0 if self.spec.terrain is None else float(self.spec.terrain.z.max())
            eye = np.array([(x0 + x1) / 2, (y0 + y1) / 2, top + height])
            target = eye - np.array([0.0, 0.0, 1.0])
        return eye, look_at_quat(eye, target)

    def render_view(self, eye, target, width=960, height=540, fov=50.0, ss=2):
        """Free camera at `eye` looking at `target` (world meters). Returns rgb uint8 HxWx3."""
        key = ("view", width, height, fov, ss)
        if key not in self.cam_buffers:
            u = self.camera.utils
            self.cam_buffers[key] = {
                "ss": ss,
                "rays": u.compute_camera_rays_pinhole(width * ss, height * ss, camera_fovs=math.radians(fov)),
                "color": u.create_color_image_output(width * ss, height * ss, 1),
                "depth": u.create_depth_image_output(width * ss, height * ss, 1),
            }
        return self._render_buf(self.cam_buffers[key], width, height, np.asarray(eye, float),
                                look_at_quat(eye, target))[0]

    def render(self, name):
        """Returns (rgb uint8 HxWx3, depth float32 HxW meters, -1 = no hit)."""
        car = self.state_0.body_q.numpy()[self.car]
        eye, quat = self._camera_pose(name, car)
        w, h, _ = CAMERAS[name]
        return self._render_buf(self.cam_buffers[name], w, h, eye, quat)

    def _render_buf(self, buf, w, h, eye, quat):
        tf = wp.array([[wp.transformf(wp.vec3f(*eye), wp.quatf(*quat))]], dtype=wp.transformf, device=self.device)
        self.model.bvh_refit_shapes(self.state_0)
        self.camera.update(
            self.state_0, tf, buf["rays"], color_image=buf["color"], depth_image=buf["depth"], clear_data=self.clear,
        )
        ss = buf["ss"]
        rgb = buf["color"].numpy().view(np.uint8).reshape(h * ss, w * ss, 4)[..., :3]
        depth = buf["depth"].numpy().reshape(h * ss, w * ss)
        if ss > 1:
            rgb = rgb.reshape(h, ss, w, ss, 3).mean(axis=(1, 3)).astype(np.uint8)
            depth = depth[ss // 2::ss, ss // 2::ss]  # nearest sample: averaging would blur edges into fake depths
        return np.ascontiguousarray(rgb), np.ascontiguousarray(depth)

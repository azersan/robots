# Driving the sim: guide for agents

> **Playing the robot (camera only, no positions)? Use [ROBOT.md](ROBOT.md) and port 8643
> instead.** This file documents the full sim/admin API on port 8642. That API includes
> ground truth (exact poses of the robot and bins), so it's meant for tools, scoring and
> the viewer, or for tasks where using ground truth is explicitly allowed.

Read this if you are a program or AI agent that wants full access to the simulated
robot. You don't need to read the sim's source. Everything goes through HTTP.

## What you're controlling

- **Robot:** a small differential-drive tug, about 19 kg and 0.6 × 0.4 m, with two drive
  wheels, a front caster, a front camera, and a tow hook on its nose.
- **World:** a simulated yard with wheelie bins (a blue 96-gal recycling cart and a gray
  64-gal trash cart).
- **Typical job:** drive to a bin, line the hook up with the bin's latch bar, latch, and
  tow.

Physics and cameras run on a GPU (Newton / MuJoCo Warp) behind a FastAPI gateway.
There is one shared sim. The web viewer and any other clients see and affect the same
world, so coordinate if more than one controller is running.

## Connecting

| From | Base URL |
|---|---|
| The Windows PC or inside WSL | `http://localhost:8642` |
| Anywhere on the tailnet (phone, other machines) | `https://aazersky-nzxt.tail3f88ef.ts.net:8443` |

Quick health check: `GET /api/info` should return JSON. If nothing answers, the gateway isn't
running. Start it from PowerShell on the PC:

```powershell
wsl -d Ubuntu-24.04 -- bash /mnt/c/Users/antho/OneDrive/Documents/GitHub/robots/sim-playground/run_gateway.sh --scene anna_pl
```

Interactive API docs (auto-generated): `/docs`. Machine-readable schema: `/openapi.json`.
POST bodies are JSON (`Content-Type: application/json`). Unknown fields are ignored and
missing fields take their defaults.

## Conventions

- **Units:** meters, seconds, radians. Velocities are in m/s and rad/s.
- **World frame:** right-handed, **Z up**, X east, Y north. Ground is z = 0 and is flat for
  now.
- **Yaw:** rotation about +Z, measured from +X, counter-clockwise positive, wrapped to
  (−π, π]. A yaw of π/2 faces north.
- **Robot body frame:** X forward, Y left, Z up, origin at the chassis center (z ≈ 0.12 m
  when sitting on the ground).
- **Bin body frame:** +X points out of the bin's **back**, the vertical side with the
  wheels and handle. A bin's `yaw` is the direction its back faces. The latch bar sits
  0.06 m behind the back face, 0.08 m off the ground, between the wheels. The robot docks
  on the back: it ends up facing the bin, with its own yaw ≈ the bin's yaw + π.
- **Time:** the sim advances in frames of 1/60 s. `time` in the state is sim seconds since
  the last reset.

## Time: real time vs lockstep

The sim is always in one of two modes. `GET /api/info` reports which one in `running`.

| Mode | How time moves | Use it for |
|---|---|---|
| Real time (`running: true`) | The gateway steps the sim itself at 60 fps × `speed` | Humans driving, live demos, controllers that run on wall-clock time |
| Lockstep (`running: false`) | Time only moves when you call `POST /api/step` | Scripted or AI controllers: deterministic, and as fast as the sim allows (several times real time) |

Switch modes with `POST /api/run {"running": false}` (lockstep) or
`POST /api/run {"running": true, "speed": 1.0}` (real time; `speed` from 0.05 to 20).

`/api/step` returns **409** while the sim is running in real time, so switch to lockstep
first.

When you finish, leave the sim the way you found it: switch back to real time if a person
might be watching. Touching the drive pad in the viewer also switches to real time.

**Typical lockstep loop (20 Hz control):**
```
POST /api/run {"running": false}
loop:
    state = POST /api/step {"frames": 3}   # 3 frames = 0.05 s; the reply is the new state
    decide (v, w) from state / camera
    POST /api/drive {"v": ..., "w": ...}
```

## Endpoints

### `GET /api/info`
Static facts about the current scene. Read it once after connecting and again after any
scene change.
```json
{
  "scene": "anna_pl", "scenes": ["flat", "anna_pl"],
  "cameras": {"front": {"width": 320, "height": 240, "fov": 70.0},
              "chase": {"width": 480, "height": 320, "fov": 60.0},
              "overhead": {"width": 512, "height": 512, "fov": 50.0}},
  "front_mount": {"pos": [0.31, 0.0, 0.12], "dir": [1.0, 0.0, -0.12], "chassis_z": 0.12},
  "route": [[0.0, 0.0], [3.4, -1.9], "..."],
  "bounds": [xmin, ymin, xmax, ymax],
  "fps": 60,
  "car": {"wheel_radius": 0.1, "track": 0.48, "max_wheel_speed": 2.0, "hook_local": [0.34, 0.0, -0.04]},
  "bin": {"types": {"recycling": {"depth": 0.86, "width": 0.67, "height": 1.1},
                    "trash": {"depth": 0.74, "width": 0.61, "height": 1.0}},
          "latch_out": 0.06, "latch_z": 0.08, "latch_range": 0.08},
  "running": true, "speed": 1.0, "cmd_timeout": 1.0
}
```
- `route` is a reference path (world XY polyline) from the start to the bins, for scenes
  that have one. On `anna_pl` it's the ~91 m driveway route; treat it like a GPS track
  someone drew.
- `fov` is the **vertical** field of view in degrees.
- `hook_local` is the hook point in the robot body frame.

### `GET /api/state`
Everything that changes. Cheap to call (no rendering).
```json
{
  "scene": "flat", "seed": 6, "time": 19.62, "frame": 1177,
  "car": {"x": 6.18, "y": 0.30, "z": 0.12, "yaw": 0.119,
          "v": -0.002, "w": -0.0002, "upright": 1.0,
          "hook": [6.521, 0.341, 0.080],
          "cmd": {"v": 0.0, "w": 0.0}},
  "bins": [{"id": 0, "kind": "recycling", "x": 7.01, "y": 0.40, "yaw": -3.02, "upright": 1.0,
            "latch": [6.519, 0.340, 0.080]},
           {"id": 1, "kind": "trash", "...": "..."}],
  "latched": 0,
  "running": false, "sim_fps": 35.5
}
```
- `car.v` is the forward speed and `car.w` the yaw rate, both measured.
- `car.cmd` is the last command the robot is executing.
- `upright` is the z component of the body's up vector: 1 is level, below ~0.7 means
  tipped over.
- `hook` is the hook point in world coordinates.
- `bins[].latch` is the world position of each bin's latch bar.
- `latched` is the bin id the hitch is attached to, or `null`.

**About ground truth:** `state` contains exact poses for everything, including the bins.
That's fine for debugging and scoring. If your task is to *find* bins with sensors, don't
read `bins` for control; use the cameras. The car's own pose (`car.x/y/yaw`) is treated
as known, standing in for GPS/odometry.

### `POST /api/drive`
```json
{"v": 0.8, "w": 0.3}
```
- `v` is forward speed in m/s (negative to reverse). `w` is yaw rate in rad/s (positive
  turns left).
- Internally this becomes wheel speeds `v ∓ w·track/2`. If either wheel would exceed
  `max_wheel_speed` (2 m/s), **both are scaled down together**, which keeps the arc shape.
- The wheels have finite torque. The robot reaches 1 m/s from rest in a fraction of a second,
  more slowly when towing.
- **Watchdog:** if no drive command arrives for `cmd_timeout` sim seconds (default 1.0),
  the motors stop. Re-send your command at least every ~0.5 s, even if it hasn't changed.
  The watchdog runs on sim time, so in lockstep it only fires when you step.
- Reply: `{"left": m/s, "right": m/s}`, the wheel surface speeds after scaling.

### `POST /api/latch`
```json
{"engage": true}
```
- **Engage:** attaches the hitch if the hook is within `latch_range` (0.08 m, 3D distance)
  of any bin's latch bar.
  - Success: `{"latched": 0, "nearest": {"bin": 0, "distance": 0.002}}`.
  - Too far: `{"latched": null, "nearest": {...}}`. Use `nearest.distance` to see how far
    off you were.
- **Release:** `{"engage": false}` replies `{"latched": null}`.
- The hitch is a stiff spring-damper between the hook and the bar, like a short strap.
  It transmits pull, push and turning, so the bin follows the robot.
- Stop before engaging. Latching while moving fast jerks the bin.

### `POST /api/reset`
```json
{"scene": "anna_pl", "randomize": true, "seed": 3}
```
- **All fields are optional.** No `scene` keeps the current one.
- **`randomize: true`** jitters the robot start (±0.5 m, ±15°) and each bin (±0.25 m along,
  ±0.15 m across, ±20° yaw), seeded by `seed`, so the same seed gives the same layout.
- **Unhooks and stops:** clears the hitch, stops the motors, and sets `time` to 0.
- **Changing scene rebuilds the world:** about 2–5 s, longer on the very first build after
  an update while GPU kernels compile. Re-read `/api/info` afterwards.
- **Reply:** the new state.

### `POST /api/step` (lockstep only)
`{"frames": 3}`: advance by N frames (0–6000; 60 = 1 s). The reply is the new state.

### `POST /api/run`
`{"running": true, "speed": 2.0}` or `{"running": false}`. Reply: `{"running": ..., "speed": ...}`.

### `GET /api/episode`
The ground-truth log since the last reset, one row every 0.1 s:
`{"columns": ["t", "x", "y", "yaw", "latched", "bin0_x", "bin0_y", "bin0_upright", ...], "rows": [...], "start": [x, y, yaw]}`.
`latched` is −1 when nothing is hooked. Use it to score runs, including runs driven through
the robot API.

### `POST /api/config`
`{"cmd_timeout": 0.5}` sets the watchdog in sim seconds (0 disables it; don't disable it
while people are driving). Reply: `{"cmd_timeout": ...}`.

### Cameras
- `GET /api/camera/{name}.jpg` (add `?quality=1-95`) or `.png`: an RGB frame from
  `front`, `chase`, or `overhead`.
- `GET /api/camera/{name}/depth.png`: a 16-bit grayscale PNG where each pixel is the
  **distance along the ray in millimeters** (not z-depth). 0 means no hit (sky).

What each camera is for:
- **`front`** (320×240, 70° vertical FOV): the robot's own sensor, mounted on the nose
  0.24 m off the ground and pitched down about 7°. Use this one for perception.
- **`chase`** (480×320): a third-person view from behind and above the robot. It's for
  humans; don't use it for control.
- **`overhead`** (512×512): straight down over the whole scene, north up. Also for humans.

Each image is rendered when you request it: roughly 5–20 ms, more for large frames. Fetch
only what you need. ~10 Hz for the front camera is plenty for most controllers.

## Turning front-camera depth into world points

The cameras use the OpenGL convention: the camera looks down its −Z axis, +Y is image-up,
+X is image-right. Pixel (u, v) has its origin at the top-left. With `W, H, fov` from
`/api/info`:

```python
import math, io, numpy as np
from PIL import Image

def front_depth_to_world(png_bytes, car, info):
    cam, mount = info["cameras"]["front"], info["front_mount"]
    W, H = cam["width"], cam["height"]
    f = (H / 2) / math.tan(math.radians(cam["fov"]) / 2)      # focal length in pixels (square pixels)
    d = np.asarray(Image.open(io.BytesIO(png_bytes)), np.float32) / 1000.0   # meters along the ray
    u = (np.arange(W) + 0.5 - W / 2) / f
    v = -(np.arange(H) + 0.5 - H / 2) / f
    uu, vv = np.meshgrid(u, v)
    rays = np.stack([uu, vv, -np.ones_like(uu)], -1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    valid = d > 0
    p_cam = rays[valid] * d[valid, None]

    # Camera -> robot body: columns are the camera's +X (right), +Y (up), +Z (back) in body coords.
    fwd = np.asarray(mount["dir"], float); fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    R_bc = np.stack([right, up, -fwd], axis=1)
    p_body = p_cam @ R_bc.T + np.asarray(mount["pos"])

    # Body -> world (the ground is flat, so the robot is level).
    c, s = math.cos(car["yaw"]), math.sin(car["yaw"])
    R_wb = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return p_body @ R_wb.T + [car["x"], car["y"], mount["chassis_z"]]
```

To sanity-check it, points on open pavement should come out at z ≈ 0.005. A working,
tested version of this, plus a bin detector that fits the known cart footprints, is in
[`controllers/perception.py`](controllers/perception.py).

## Docking geometry (how to hook a bin)

Given a bin's latch point `L` (world xy) and its yaw `ψ` (back-face normal `n = (cos ψ, sin ψ)`):

- **Dock pose:** robot center at `L + n · hook_local.x` (≈ 0.34 m), facing `ψ + π`. At
  that pose the hook sits on the bar.
- **Pre-dock pose:** about 1.2 m further out along `n`, facing the same way. Get there
  first, turn in place to face the bin, then creep straight in at ≤ 0.25 m/s.
- **Engage:** when the hook is within a few cm of the bar, stop, step a few frames, and
  call `/api/latch`.

At the dock pose the robot's nose is about 2 cm from the bin. The bin is solid, so
driving into it pushes it away.

## Physical limits and quirks

- **Speeds:** up to about 2 m/s per wheel. The reference controller cruises at 1.2 m/s and
  docks at 0.25 m/s.
- **Turning in place** works well (`v = 0`). The caster is a free ball.
- **Towing:** an empty cart drags at about 30 N of resistance, well within the robot's
  traction. Full carts aren't modeled yet.
- **Static obstacles:** tree trunks and houses are solid. Leaves, pavement and lawn
  markings are visual only. Collisions with them haven't been characterized, so avoid them.
- **Ground:** flat everywhere for now.

## Existing code you can reuse

- [`controllers/common.py`](controllers/common.py): a thin `Gateway` client (`state()`,
  `drive()`, `step()`, `latch()`, `camera()`, `depth()`, ...) plus pure-pursuit helpers.
- [`controllers/fetch_bin.py`](controllers/fetch_bin.py): the complete reference behavior
  (follow the route, find a bin with the camera, align, dock, latch, tug), with a
  per-trial report.

  ```bash
  ~/venvs/simpg/bin/python controllers/fetch_bin.py --scene anna_pl --sense camera --trials 5 --seed 1
  ```
- [`controllers/perception.py`](controllers/perception.py): depth to world points, and the
  bin finder.

## Etiquette

- Assume a human may be watching in the viewer at `/`, possibly from a phone. Don't leave
  the sim paused or the robot driving when you finish.
- Prefer lockstep for anything long-running or repeated. It doesn't depend on network
  timing and runs faster than real time.
- `/api/reset` and scene changes affect everyone using the sim.

# Robot API: camera-only control

This is the interface for an agent that drives the robot **the way the real one would**:
it sees through two color cameras (one facing forward, one facing backward), has a
compass, can drive and latch, and gets plain-language directions. It does **not** know where it is, where the bins are, or
what the map looks like. It has to work all of that out from the picture.

(There's also a full sim/admin API with ground truth on port 8642, documented in
[AGENTS.md](AGENTS.md). It's for tools, scoring and the viewer. Don't use it when you're
playing the robot.)

## Connecting

| | |
|---|---|
| Base URL | `http://localhost:8643` (on the Windows PC or inside WSL) |
| Interactive docs | `http://localhost:8643/docs` |
| Start here | `GET /robot/info`: your directions, camera specs, and limits |

POST bodies are JSON (`Content-Type: application/json`). A person may be watching the robot
live in the viewer on port 8642, possibly from a phone.

## What you have

| You get | Details |
|---|---|
| **Front camera** | Color only. 320×240. Mounted on the nose, 0.24 m off the ground, pitched down ~7°. |
| **Rear camera** | Same specs, on the tail, looking backward. It's what you see where you're going when reversing. |
| **Compass** | Heading in degrees: 0 = north, 90 = east (clockwise). About ±2° of noise. |
| **Drive** | Forward speed `v` (m/s) and turn rate `w` (rad/s, positive = left). |
| **Latch** | A hook on the nose. `engage` answers only yes or no, like a limit switch. |
| **Directions** | Plain text in `/robot/info`, like what a person would tell you. |
| **Sim time** | Seconds since the last reset. |

What you **don't** have: position, a map, depth, other cameras, wheel odometry, or the
bins' locations.

## Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/robot/info` | | Directions, camera specs, drive limits, `cmd_timeout`, `running` |
| GET | `/robot/sensors` | | `{"time", "compass_deg", "latched", "cmd", "running"}` |
| GET | `/robot/camera/front.jpg`, `/robot/camera/rear.jpg` (or `.png`) | | The current image from that camera. Optional `?quality=1-95` for JPEG. `/robot/camera.jpg` is the same as the front one. |
| POST | `/robot/drive` | `{"v": 0.8, "w": 0.2}` | Sensors |
| POST | `/robot/latch` | `{"engage": true}` or `{"engage": false}` | `{"latched": true/false}` |
| POST | `/robot/step` | `{"frames": 3}` | Sensors, after advancing time (lockstep only) |
| POST | `/robot/run` | `{"running": false}` or `{"running": true, "speed": 1.0}` | `{"running", "speed"}` |
| POST | `/robot/reset` | `{"seed": 3, "randomize": true}` (and optionally `"scene"`) | Sensors |

- **`reset`** puts you back at the start. `randomize` (default `true`) jitters your start
  pose and the bins a little, seeded by `seed`.
- **Scenes:** `anna_pl` is the real job (a ~90 m driveway run). `flat` is a short concrete
  pad with the bins straight ahead, good for practicing the dock.

## Time: lockstep vs real time

The sim is in one of two modes (`running` in `/robot/info` and in the sensors):

- **Lockstep** (`POST /robot/run {"running": false}`): time moves only when you call
  `/robot/step`. One frame is 1/60 s. Use this for autonomous runs. It's deterministic and
  faster than real time.
- **Real time** (`{"running": true}`): the sim advances by itself. This is for people
  watching or driving by hand. `/robot/step` returns 409 in this mode.

A typical loop at 10–20 Hz:
```
POST /robot/run {"running": false}
loop:
    img = GET /robot/camera.jpg
    decide (v, w) from the image (and compass)
    POST /robot/drive {"v": ..., "w": ...}
    POST /robot/step {"frames": 3}      # 0.05 s
```

**Watchdog:** if no drive command arrives for `cmd_timeout` sim seconds (1.0), the motors
stop. Send a drive command every tick, even if it hasn't changed.

When you're done, stop the robot and switch back to real time
(`POST /robot/run {"running": true}`), so anyone watching sees a live sim.

## Driving

- The robot is a differential-drive tug: two drive wheels at the back, a free ball caster
  at the front, about 0.6 m long and 0.4 m wide. It can **turn in place** (`v = 0`).
- It pivots about the middle of the drive axle, 0.10 m behind its center. That puts the
  front camera 0.41 m ahead of the pivot and the rear camera 0.21 m behind it. Both are
  given as `from_turn_center_m` in `/robot/info`.
- Wheel speed is capped at 2 m/s. If a command would exceed it, both wheels are scaled
  down together, so the curve keeps its shape. Sensible speeds are 0.8–1.2 m/s cruising,
  ≤ 0.25 m/s near a bin, and ≤ 0.5 m/s while towing.
- The compass counts clockwise, so turning **left** (`w > 0`) makes `compass_deg` go
  **down**.
- The ground is flat. Tree trunks and the houses are solid; don't hit them.

## What the world looks like

- **Pavement:**
  - The house driveway is light gray-beige concrete with faint brush marks.
  - The lane (Anna Pl) and the street are darker gray speckled asphalt.
- **Grass** is olive-green lawn texture. Wooded strips are grass with tall trees. Stay off
  all of it.
- **Bins:**
  - Recycling is a big blue 96-gal cart.
  - Trash is a smaller gray 64-gal cart.

  Both have a lid, a handle across the top of the back, two black wheels at the bottom of
  the back, and a **yellow latch bar** low between the wheels.
- **Lighting:** afternoon sun from the southwest, with soft shadows. Shadows of the trees
  and houses fall across the pavement and the lawn.

Pavement vs grass by color is the main way to keep on the road. Gray (low saturation) means
pavement; green means grass. Shadows darken both without changing which is which.

Beige concrete can come close to a naive "green" test. Check that green beats red, not
just blue. In shadow, concrete and asphalt look alike, and the concrete-to-asphalt seam
where the driveway meets the lane shows up as a darker band rather than a sharp edge.

## Camera geometry (for judging distance)

Both cameras have the same lens and mounting height; the rear one simply faces backward.
The formulas below work for either. For the rear camera, "forward" means behind the robot
and "left" means the robot's right, since that's the camera's own left.

- **Field of view:** 70° vertical, ~86° horizontal, square pixels.
- **Focal length:** `f = (240/2) / tan(35°) ≈ 171.4` px.
- **Mounting:** the camera sits 0.24 m above flat ground and is pitched down 6.8°.
- **Visible ground:** the horizon is about 20 px above the image center. The bottom edge
  of the image sees the ground about 0.27 m ahead of the camera.

For a pixel (u, v), with the origin at the top-left and the point on the ground:

```python
import math
W, H, f, h, pitch = 320, 240, 171.4, 0.24, math.radians(6.8)

def ground_point(u, v, h=h):
    """(forward_m, left_m) from the camera to the point at height 0.24 - h seen at pixel (u, v).
    With the default h, that point is on the ground. Returns None at or above the horizon."""
    x = (u + 0.5 - W / 2) / f                  # image right, in units of focal length
    y = -(v + 0.5 - H / 2) / f                 # image up
    # Ray in the robot frame (forward, left, up), camera pitched down by `pitch`.
    fwd_c = y * math.sin(pitch) + math.cos(pitch)
    up_c = y * math.cos(pitch) - math.sin(pitch)
    if up_c >= 0:
        return None
    t = h / -up_c
    return t * fwd_c, -t * x
```

For a point that isn't on the ground, pass `h = 0.24 − height`. For example, the latch bar
is 0.08 m up, so use `ground_point(u, v, h=0.16)`. The bins have known sizes
(see the latching section), so their apparent width also gives range: at distance `d`, a
0.67 m-wide blue bin is about `0.67 · f / d` px wide.

## How to latch onto a bin

The hook is on the robot's **nose**, at 0.08 m height, 0.03 m ahead of the camera. It
grabs the bin's **yellow latch bar**, which runs between the two back wheels:

- **Bar position:** 0.08 m off the ground and 0.06 m behind the bin's back face.
- **Bar size:** about 0.24 m wide, with two short yellow brackets holding it off the bin.
- **Engage range:** the latch engages if the hook is within **0.08 m** of the bar.

The procedure:

1. **Come at the bin from its back:** the side with the handle across the top and the two
   black wheels at the bottom. You can't latch from the front (lid side) or the sides.
2. **Stop about 1–1.5 m behind the bin and line up.** On `anna_pl` this is the tightest
   spot on the route. The carts stand on the street with their backs to the lane mouth,
   and there are only **about 2.7 m (blue) and 3.0 m (gray) of pavement** between a cart's
   back and the lawn. Line up close to the cart, and measure the lawn edge before
   maneuvering. Turn so the bin is centered in the
   image and its back face looks square: the left and right edges, and the two wheels,
   look symmetric. The yellow bar should sit horizontally in the middle, between the
   wheels.
3. **Creep straight in** at ≤ 0.2 m/s, steering gently to keep the bar centered
   left-to-right. Approach **straight on**. Coming in at an angle or off-center by more
   than a few centimeters misses the bar. Bumping the bin's body pushes it away.
4. **Expect a blind spot at the end.** The bar stays in view until the hook is about
   0.12–0.15 m from it. After that it drops below the bottom edge of the image, so you
   cover the last few centimeters blind. Commit to your line-up before that point.
5. **Stop, then call latch:** `POST /robot/latch {"engage": true}`. Or creep in 1–2 cm at
   a time, calling latch after each nudge. If you get `{"latched": false}` after you think
   you've arrived, back straight out about 1 m, re-center, and try again.

What the bar looks like from the robot at different hook-to-bar distances (blue bin,
centered and square):

| Distance | Image | What you see |
|---|---|---|
| 3 m | [docs/robot_view_300cm.jpg](docs/robot_view_300cm.jpg) | Both bins, small, at the end of the pavement. This view is taken from the lawn behind the strip (on `anna_pl` you can't stand 3 m back on pavement), so expect more pavement in the foreground from the lane. |
| 1 m | [docs/robot_view_100cm.jpg](docs/robot_view_100cm.jpg) | Bin fills the middle; the wheels and the yellow bar are visible at the bottom of it. |
| 0.5 m | [docs/robot_view_050cm.jpg](docs/robot_view_050cm.jpg) | Bin fills most of the frame, with the wheels on either side; the bar is just below the middle of the image. |
| 0.3 m | [docs/robot_view_030cm.jpg](docs/robot_view_030cm.jpg) | The bar and its brackets are large in the lower middle. |
| 0.15 m | [docs/robot_view_015cm.jpg](docs/robot_view_015cm.jpg) | The bar is at the very bottom edge. |
| 0.10 m | [docs/robot_view_010cm.jpg](docs/robot_view_010cm.jpg) | Only the bracket corners are visible; the bar itself is out of view. |

Bin sizes:

| Bin | Color | Back width | Depth | Height |
|---|---|---|---|---|
| Recycling | blue | 0.67 m | 0.86 m | 1.10 m |
| Trash | gray | 0.61 m | 0.74 m | 1.00 m |

## After you latch: towing

- The hitch is a short, stiff link. Once latched, the bin moves with your nose, staying in
  front of you and facing you. The robot can pull it (by reversing), push it, and turn
  with it.
- An empty cart takes about 30 N to drag, well within what the robot can pull.
- `{"engage": false}` releases it.
- While latched, the bin fills the **front** camera. Use the **rear** camera to see where
  you're going when you reverse with the bin in tow:
  - [docs/robot_view_docked_front.jpg](docs/robot_view_docked_front.jpg): the front
    camera when docked, all bin.
  - [docs/robot_view_docked_rear.jpg](docs/robot_view_docked_rear.jpg): the rear camera
    at the same moment, looking up the lane.
- Turning in place with a bin attached swings the bin around in a ~1 m arc, so make sure
  there's room. The two carts stand side by side about 2 m apart (center to center), so
  pivoting right next to them can swing the towed cart into the other one. Back straight
  out first, then turn.

## How runs are judged

The sim records the true trajectory, which you can't see. A run is scored on:

- **Bin home:** the bin is still latched at the end, upright, with its center within 3 m
  of where the robot started.
- **Pavement:** time spent with any wheel on grass. Less is better; zero is the goal.
- **Time:** sim seconds, including the time used for re-tries.

"Home" means back in front of the garage, where the driveway starts. Recognize it from the
camera: the end of the concrete driveway, with the house's garage doors.

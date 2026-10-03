# Sim Playground

A simulated robot world that other programs drive through an HTTP API, the same way
they'd drive real hardware. Physics and cameras come from [Newton](https://github.com/newton-physics/newton)
(MuJoCo Warp + a Warp raytracer) on the RTX 5090; everything runs in WSL2 Ubuntu.

Current scenario: a small tug drives from the garage at #11 down Anna Pl (~91 m), finds the
trash bins at the street, docks on one, and latches on.

**Controlling the sim from another program or agent? Read [AGENTS.md](AGENTS.md).** It covers
the full API, units and frames, timing modes, camera math, and docking geometry.

```
 controllers/ (any HTTP client: scripts, policies, Claude later)
        │  HTTP
        ▼
 simpg/gateway.py   FastAPI: /api/drive, /api/latch, /api/state, /api/camera/...
        │
 simpg/sim.py       Newton model: car, bins, scenes, cameras, tow hitch
        │
 Newton 1.6 (SolverMuJoCo + SensorTiledCamera) on CUDA
```

## Setup (once)

From Windows PowerShell:

```powershell
wsl -d Ubuntu-24.04 -- bash /mnt/c/Users/antho/OneDrive/Documents/GitHub/robots/sim-playground/setup_wsl.sh
```

Then fetch the CC0 photo textures (grass, concrete, asphalt, bark from ambientCG) into the
git-ignored `assets/` folder:

```powershell
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/c/Users/antho/OneDrive/Documents/GitHub/robots/sim-playground && ~/venvs/simpg/bin/python tools/fetch_assets.py"
```

The scenes still work without them, using flat colors. The venv lives at `~/venvs/simpg`
on the Linux side (Python on `/mnt/c` is slow). The first run compiles GPU kernels for a
minute or two; after that they're cached and a scene builds in ~2–5 s.

## Run

```powershell
wsl -d Ubuntu-24.04 -- bash /mnt/c/Users/antho/OneDrive/Documents/GitHub/robots/sim-playground/run_gateway.sh --scene anna_pl
```

Open http://localhost:8642/ for the viewer (chase + front cameras, map, state). Drive with
the on-screen pad (drag up = forward, sideways = turn; release = stop) or, on a keyboard,
W/A/S/D, Space to stop, L to latch. Touching the pad switches the sim to real time.

**From your phone (Tailscale):** https://aazersky-nzxt.tail3f88ef.ts.net:8443/ is proxied to
the gateway by `tailscale serve` (tailnet only). Set up once with
`tailscale serve --bg --https=8443 http://127.0.0.1:8642`; remove with
`tailscale serve --https=8443 off`. It's needed because WSL's NAT networking only exposes
the gateway on Windows localhost.

Run the autonomous fetch (from WSL, in `sim-playground/`):

```bash
~/venvs/simpg/bin/python controllers/fetch_bin.py --scene anna_pl --sense camera --trials 5 --seed 1
```

`--sense truth` reads bin poses from the sim; `--sense camera` finds them in the front depth
camera. In both, the car's own pose comes from the sim (standing in for GPS/odometry).

## API

| Method | Path | Body / notes |
|---|---|---|
| GET | `/api/info` | scenes, cameras (+ front mount), route, car/bin dimensions |
| GET | `/api/state` | car pose/velocity, bins, latch status, sim time |
| POST | `/api/drive` | `{"v": m/s, "w": rad/s}`; motors stop after `cmd_timeout` (1 s) without a new command |
| POST | `/api/latch` | `{"engage": true}` latches if the hook is within 8 cm of a bin's latch bar |
| POST | `/api/reset` | `{"scene": "anna_pl", "randomize": true, "seed": 3}` |
| POST | `/api/run` | `{"running": false}` for lockstep, `{"running": true, "speed": 2}` for real time |
| POST | `/api/step` | `{"frames": n}` (lockstep only; 60 frames = 1 s) |
| POST | `/api/config` | `{"cmd_timeout": seconds}` (0 disables the watchdog) |
| GET | `/api/camera/{front,chase,overhead}.{jpg,png}` | RGB frame |
| GET | `/api/camera/{name}/depth.png` | 16-bit PNG, millimeters, 0 = no hit |

```bash
curl -s localhost:8642/api/state
curl -s -o front.jpg localhost:8642/api/camera/front.jpg
curl -s -X POST localhost:8642/api/drive -H 'content-type: application/json' -d '{"v":0.5,"w":0}'
```

## Files

- `simpg/sim.py`: the Newton model (car, bins, scenes, hitch, cameras)
- `simpg/site_anna_pl.py`: the driveway layout traced from the satellite screenshot (pixel coordinates and scale)
- `simpg/gateway.py`, `simpg/viewer.html`: HTTP API and web viewer
- `controllers/fetch_bin.py`: route following, docking, latching
- `controllers/perception.py`: bin detection from the depth camera (fits the bin's known footprint)
- `simpg/meshes.py`: procedural visual meshes (carts, trees, roofs, pavement ribbons) and generated textures
- `tools/smoke.py`: build/drive/render check without the gateway
- `tools/beauty.py`: renders 1280×720 showcase frames from free cameras
- `tools/fetch_assets.py`: downloads the photo textures

## Modeling notes

- **Car:** differential-drive tug, about 19 kg, 0.6 × 0.4 m, 0.1 m wheels, 0.48 m track, free-rolling front ball
  caster, tow hook on the nose 8 cm off the ground.
- **Bins:** a blue 96-gal recycling cart (0.86 × 0.67 × 1.10 m, 15 kg) and a gray 64-gal trash cart
  (0.74 × 0.61 × 1.00 m, 12 kg).
  - Visually they're modeled carts: tapered tub, lid, handle, wheels.
  - Physically each is one sliding box with μ=0.25, standing in for a cart on its wheels.
  - The latch bar sits low on the back (wheel/handle) side, which faces the house at the curb.
- **Rendering:** Newton's Warp raytracer with photo textures on UV-mapped meshes, one directional sun with
  shadows, and hemispheric ambient light. The robot and chase cameras render at 2× and downsample, because
  the raytracer has no texture mipmaps. There are no normal or roughness maps, reflections, or bounce light.
- **Hitch:** a stiff spring-damper between the hook and the latch bar, like a strap or clamp.
- **Friction:** MuJoCo uses the larger of the two surfaces' μ, so the ground is kept low (0.2) and each object's
  own μ decides.
- **Anna Pl scene:** flat ground. The pavement is visual only; houses and tree trunks are solid. The scale
  (`M_PER_PX`) is estimated from house and road widths.

## Results so far

| Scene | Sensing | Latched | Sim time |
|---|---|---|---|
| flat (10 randomized seeds) | camera | 10/10 | ~17 s |
| anna_pl (8 randomized seeds) | camera | 8/8 | ~86 s |
| anna_pl (6 + 3 seeds) | truth | 9/9 | ~89 s |
| flat, modeled carts (6 seeds) | camera | 5/6 | ~16 s |
| anna_pl, modeled carts (5 seeds) | camera | 5/5 | ~86 s |

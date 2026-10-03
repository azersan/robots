# sim

Robot sims on the NZXT (WSL2 Ubuntu 24.04, RTX 5090, Newton 1.6 + MuJoCo Warp). Phase 1 of the Robotics Sim
Playground plan: a car on a real scanned driveway.

## Watch the driveway car from the Mac

```bash
sim/view_driveway.sh            # opens http://localhost:8080 after ~25 s; Ctrl-C stops the sim
```

A small rover (10 cm wheels, differential drive, caster) starts where the scan started and follows the path you
walked on the way out (pure pursuit), restarting from the street each time it reaches the top. It runs at about 7x
real time on the 5090.

## Setup on the NZXT (done 2026-10-03)

- WSL distro `Ubuntu-24.04`, user `tony`; venv at `~/robots/.venv` (uv, Python 3.12):
  `uv pip install "newton[sim,importers,examples]" viser==1.0.26 -r scan-to-sim/scan2sim/requirements.txt`
- Code: `~/robots/sim/`, `~/robots/scan-to-sim/scan2sim/` (copied, not a git clone yet).
- Scans: `~/scans/<name>.scan/` with `sim/` from scan2sim and `poses.json`.

New scan: compile with `python -m scan2sim <zip> --corridor 3 --ground depth`, copy `<name>.scan/sim` and
`poses.json` to `~/scans/<name>.scan/` on the NZXT, then `sim/view_driveway.sh <name>.scan`.

## Things that bit us (Newton 1.6 / mujoco-warp 3.12)

- MuJoCo Warp's own heightfield collision put this terrain's surface ~10-15 cm too high and launched the car;
  `driveway.py` uses `SolverMuJoCo(use_mujoco_contacts=False)` with Newton's `CollisionPipeline` instead.
- MuJoCo collides with a mesh's convex hull, so the scan's mesh colliders are ignored and the ground is the
  heightfield (10 cm cells; 5 cm overflows MuJoCo's per-geom contact budget).
- `add_body` adds a free joint; articulated parts use `add_link`. Velocity targets are `Control.joint_target_qd`.
- One small robot is launch-bound on the GPU: the substep loop is captured once as a CUDA graph.
- WSL stops a distro when no Windows-side session is attached, killing background processes; the sim runs in the
  foreground of the SSH session. Tunnel to `127.0.0.1`, not `localhost` (Windows resolves that to IPv6).

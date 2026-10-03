"""Milestone checks: drop a 20 cm box on the scanned ground and confirm it rests on it; tagged objects stay put."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .ground import Heightfield

BOX_HALF = 0.1


def load_heightfield(sim: Path) -> Heightfield:
    rep = json.loads((sim / "report.json").read_text())["heightfield"]
    raw = np.fromfile(sim / rep["file"], dtype="<f4")
    ny, nx = int(raw[:2].view("<i4")[0]), int(raw[:2].view("<i4")[1])
    z = raw[2:].reshape(ny, nx).astype(np.float64)
    return Heightfield(rep["x0"], rep["y0"], rep["cell"], z, np.ones_like(z, dtype=bool))


def drop_point(hf: Heightfield) -> tuple[float, float]:
    """The scan origin (0, 0) if it's on the grid, else the grid center."""
    ny, nx = hf.z.shape
    if hf.x0 <= 0 <= hf.x0 + (nx - 1) * hf.cell and hf.y0 <= 0 <= hf.y0 + (ny - 1) * hf.cell:
        return 0.0, 0.0
    return hf.x0 + (nx - 1) * hf.cell / 2, hf.y0 + (ny - 1) * hf.cell / 2


def mujoco_check(sim: Path, seconds: float = 3.0, at: tuple[float, float] | None = None) -> dict:
    import mujoco

    hf = load_heightfield(sim)
    x, y = at or drop_point(hf)
    ground = float(hf.sample(np.array(x), np.array(y)))
    xml = (sim / "scene.xml").read_text()
    probe = (f'<body name="probe_box" pos="{x} {y} {ground + 0.5}"><freejoint/>'
             f'<geom type="box" size="{BOX_HALF} {BOX_HALF} {BOX_HALF}" mass="1" rgba="0.2 0.4 0.9 1"/></body>')
    tmp = sim / "_check_scene.xml"
    tmp.write_text(xml.replace("</worldbody>", f"    {probe}\n  </worldbody>"))
    try:
        model = mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    start = {model.body(i).name: data.xpos[i].copy() for i in range(1, model.nbody)}
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)

    box_id = model.body("probe_box").id
    box_z = float(data.xpos[box_id][2])
    # On a slope the box's lowest corner touches first; allow for that tilt.
    slope = float(np.hypot(*np.gradient(hf.z, hf.cell))[
        int(np.clip((y - hf.y0) / hf.cell, 0, hf.z.shape[0] - 1)),
        int(np.clip((x - hf.x0) / hf.cell, 0, hf.z.shape[1] - 1))])
    tol = 0.02 + BOX_HALF * slope
    result = {
        "engine": "mujoco " + mujoco.__version__,
        "drop_xy": [round(x, 3), round(y, 3)],
        "ground_z": round(ground, 4),
        "box_center_z": round(box_z, 4),
        "expected_z": round(ground + BOX_HALF, 4),
        "box_rests_on_ground": abs(box_z - (ground + BOX_HALF)) <= tol,
        "box_speed": round(float(np.linalg.norm(data.qvel[model.jnt_dofadr[model.body_jntadr[box_id]]:][:3])), 4),
        "objects": {},
    }
    for i in range(1, model.nbody):
        name = model.body(i).name
        if name == "probe_box":
            continue
        moved = float(np.linalg.norm(data.xpos[i] - start[name]))
        result["objects"][name] = {"moved_m": round(moved, 4), "settled": moved < 0.05}
    result["pass"] = result["box_rests_on_ground"] and all(o["settled"] for o in result["objects"].values())
    return result


def newton_check(sim: Path, seconds: float = 2.0, at: tuple[float, float] | None = None, device: str | None = None) -> dict:
    """The same drop test in Newton, loading scene.usda through ModelBuilder.add_usd."""
    import newton
    import warp as wp

    hf = load_heightfield(sim)
    x, y = at or drop_point(hf)
    ground = float(hf.sample(np.array(x), np.array(y)))

    builder = newton.ModelBuilder()
    builder.add_usd(str(sim / "scene.usda"))
    box = builder.add_body(xform=wp.transform(wp.vec3(x, y, ground + 0.5), wp.quat_identity()), label="probe_box")
    builder.add_shape_box(box, hx=BOX_HALF, hy=BOX_HALF, hz=BOX_HALF)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverXPBD(model, iterations=10)
    s0, s1, control = model.state(), model.state(), model.control()
    start = s0.body_q.numpy()[:, :3].copy()

    dt = 1.0 / 600.0
    for _ in range(int(seconds / dt)):
        s0.clear_forces()
        contacts = model.collide(s0)
        solver.step(s0, s1, control, contacts, dt)
        s0, s1 = s1, s0

    q = s0.body_q.numpy()[:, :3]
    labels = list(builder.body_label)
    box_z = float(q[box, 2])
    result = {
        "engine": f"newton {newton.__version__} ({model.device})",
        "drop_xy": [round(x, 3), round(y, 3)],
        "ground_z": round(ground, 4),
        "box_center_z": round(box_z, 4),
        "expected_z": round(ground + BOX_HALF, 4),
        "box_rests_on_ground": abs(box_z - (ground + BOX_HALF)) <= 0.03,
        "objects": {},
    }
    for i, label in enumerate(labels):
        if i == box:
            continue
        moved = float(np.linalg.norm(q[i] - start[i]))
        result["objects"][label.rsplit("/", 1)[-1]] = {"moved_m": round(moved, 4), "settled": moved < 0.05}
    result["pass"] = result["box_rests_on_ground"] and all(o["settled"] for o in result["objects"].values())
    return result

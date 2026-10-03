"""End-to-end test on a synthetic bundle. Run: python tests/test_pipeline.py [WORK_DIR]"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_fake_bundle as fake  # noqa: E402
from scan2sim.bundle import read_glb  # noqa: E402
from scan2sim.check import load_heightfield, mujoco_check  # noqa: E402
from scan2sim.compile import compile_bundle  # noqa: E402


def main(work: Path):
    root = fake.main(work)
    report = compile_bundle(root)
    sim = root / "sim"
    failures = []

    def check(cond, msg):
        print(("PASS " if cond else "FAIL ") + msg)
        if not cond:
            failures.append(msg)

    hf = load_heightfield(sim)
    xs, ys = np.array([-3.0, 0.0, 5.0, 7.0, 2.0]), np.array([-1.0, 0.0, 6.0, 1.0, 3.0])
    err = np.abs(hf.sample(xs, ys) - fake.ground_z(xs, ys))
    check(err.max() < 0.02, f"heightfield matches the true ground (max error {err.max() * 100:.1f} cm, incl. under the barrel)")

    obj = report["objects"][0] if report["objects"] else None
    check(obj is not None and obj["name"] == "barrel", "barrel compiled as an object")
    check(abs(obj["origin"][2] - fake.ground_z(*fake.BARREL)) < 0.03, "barrel origin sits on the ground")
    check(report["background_faces"] >= 12, "wall ended up in the background")
    check(report["texture_coverage"].get("barrel", 0) > 0.4, f"barrel seen by keyframes ({report['texture_coverage']})")

    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(sim / "scene.usda"))
    visual = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Objects/barrel/Visual"))
    cols = np.array(UsdGeom.PrimvarsAPI(visual).GetPrimvar("displayColor").Get())
    pts = np.array(visual.GetPointsAttr().Get())
    side = np.abs(np.linalg.norm(pts[:, :2], axis=1) - fake.BARREL_R) < 0.02  # barrel wall vertices (object frame)
    seen = side & ~np.all(np.isclose(cols, 0.6, atol=0.01), axis=1)
    blue = seen & (cols[:, 2] > cols[:, 0] + 0.3)
    check(seen.sum() > 10 and blue.sum() / seen.sum() > 0.9,
          f"barrel vertices colored blue from keyframes ({blue.sum()}/{seen.sum()})")
    check(UsdGeom.GetStageUpAxis(stage) == "Z", "USD stage is Z-up")

    for at in [(0.0, 0.0), (0.0, 6.0), (6.0, -1.0)]:
        r = mujoco_check(sim, at=at)
        check(r["box_rests_on_ground"], f"MuJoCo box at {at} rests at z={r['box_center_z']} (expected {r['expected_z']})")
    check(r["objects"]["barrel"]["settled"], f"barrel stays put in MuJoCo (moved {r['objects']['barrel']['moved_m']} m)")

    # Walked path out and back with known drift on the return trip; corridor of 2 m around the path (x = 0).
    path_root = fake.main(work, path_mode=True)
    rep = compile_bundle(path_root, corridor=2.0, texture=False)
    drift = rep["loop_drift"]
    expected = float(np.linalg.norm(fake.DRIFT_SHIFT))
    check(drift.get("revisited", False), "path scan: return visit to the start detected")
    check(abs(drift.get("drift_m", 0) - expected) < 0.05,
          f"path scan: drift measured {drift.get('drift_m')} m (true {expected:.3f} m)")
    check(abs(abs(drift.get("yaw_deg", 0)) - fake.DRIFT_YAW_DEG) < 0.5,
          f"path scan: yaw drift measured {drift.get('yaw_deg')} deg (true {fake.DRIFT_YAW_DEG})")
    psim = path_root / "sim"
    import trimesh
    xs = []
    for name in ["ground_visual", "background"]:
        f = psim / "meshes" / f"{name}.obj"
        if f.exists():
            xs += list(trimesh.load(f, process=False).triangles_center[:, 0])
    # The corridor follows the recorded path, and the return trip's recorded path carries the planted drift.
    check(len(xs) > 0 and max(abs(x) for x in xs) <= 2.0 + np.linalg.norm(fake.DRIFT_SHIFT) + 0.05,
          f"corridor 2 m keeps only faces centered near the path (max |x| = {max(abs(x) for x in xs):.2f} m)")

    print(f"\n{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
    sys.exit(main(work))

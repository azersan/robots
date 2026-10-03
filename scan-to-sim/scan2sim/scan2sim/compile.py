"""The scan2sim pipeline: capture bundle -> <bundle>/sim/{scene.usda, scene.xml, collision/, meshes/, report.json}."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import write_mjcf, write_usd
from .bundle import Part, load, merge_parts
from .clean import clean
from .drift import loop_drift
from .collision import body_origin, hull_volume, object_collision, write_obj
from .ground import fit_heightfield, split_ground, submesh
from .texture import colorize


def within_corridor(part: Part, keyframes, width: float) -> np.ndarray:
    """Face mask: centroid within `width` meters (horizontally) of the walked path."""
    from scipy.spatial import cKDTree

    path = np.array([k.cam_to_world[:3, 3] for k in keyframes])[:, :2]
    # Densify so gaps between keyframes (up to ~0.25 m, more when tracking was lost) don't pinch the corridor.
    pts = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / 0.1)))
        pts.extend(a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:])
    dist, _ = cKDTree(np.array(pts)).query(part.vertices[part.faces].mean(axis=1)[:, :2])
    return dist <= width


def tapped_component(part: Part) -> np.ndarray:
    """Face mask of the connected component nearest the tag center (sim frame, from the manifest)."""
    import trimesh

    center = part.meta.get("tag_center")
    if center is None or len(part.faces) == 0:
        return np.ones(len(part.faces), dtype=bool)
    mesh = trimesh.Trimesh(part.vertices, part.faces, process=False)
    labels = trimesh.graph.connected_component_labels(mesh.face_adjacency, node_count=len(part.faces))
    nearest = np.argmin(np.linalg.norm(mesh.triangles_center - np.asarray(center), axis=1))
    return labels == labels[nearest]


def compile_bundle(path: Path, out: Path | None = None, cell: float = 0.05, masses: dict[str, float] | None = None,
                   default_mass: float = 5.0, friction: float = 0.8, collision: str = "auto",
                   texture: bool = True, min_confidence: int = 1, max_faces: int | None = None,
                   min_object_faces: int = 100, corridor: float | None = None) -> dict:
    bundle = load(path)
    sim = Path(out) if out else bundle.root / "sim"
    (sim / "collision").mkdir(parents=True, exist_ok=True)
    masses = masses or {}
    print(f"Bundle {bundle.root.name}: {sum(len(p.faces) for p in bundle.mesh_parts)} mesh faces, "
          f"{len(bundle.objects)} objects, {len(bundle.keyframes)} keyframes")

    drift = loop_drift(bundle.keyframes) if bundle.keyframes else {"revisited": False, "note": "no keyframes"}
    if drift.get("revisited") and "drift_m" in drift:
        print(f"Loop drift: {drift['drift_m'] * 100:.0f} cm over {drift['path_length_m']:.0f} m walked "
              f"({drift['drift_percent_of_path']}%), yaw {drift['yaw_deg']} deg: {drift['verdict']}")
    else:
        print(f"Loop drift: not measured ({drift.get('note')})")

    print("Cleaning meshes")
    scene = merge_parts(bundle.mesh_parts, "scene")
    if corridor:
        if not bundle.keyframes:
            raise SystemExit("--corridor needs keyframes (the walked path comes from their camera positions)")
        keep = within_corridor(scene, bundle.keyframes, corridor)
        print(f"Corridor {corridor} m: keeping {keep.mean() * 100:.0f}% of faces")
        scene = submesh(scene, keep, "scene")
    scene = clean(scene, max_faces=max_faces)
    objects = [clean(o, min_component_faces=10) for o in bundle.objects]
    objects = [o for o in objects if len(o.faces)]

    print(f"Fitting ground heightfield ({cell * 100:.0f} cm cells)")
    hf = fit_heightfield([scene], cell=cell)
    on_ground, rest = split_ground(scene, hf)
    ground_visual = submesh(scene, on_ground, "ground")
    background = submesh(scene, rest, "background")
    # The tag sphere usually grabs some ground around the object's base; give that back to the ground.
    objects = [submesh(o, split_ground(o, hf)[1], o.name) for o in objects]
    # The tag sphere also catches pieces of neighboring things (a chair back next to a table). Keep only
    # the connected piece at the tap point; the rest goes back to the background.
    strays = []
    for i, o in enumerate(objects):
        keep = tapped_component(o)
        if not keep.all():
            strays.append(submesh(o, ~keep, "stray"))
            objects[i] = submesh(o, keep, o.name)
    if corridor:
        strays = [submesh(st, within_corridor(st, bundle.keyframes, corridor), "stray") for st in strays]
    strays = [st for st in strays if len(st.faces)]
    if strays:
        background = merge_parts([background, *strays], "background")
    # A tag that caught almost nothing (usually a second tap on an object already tagged) would become a
    # sliver hull overlapping its neighbor and push it around in the sim.
    for o in objects:
        if len(o.faces) < min_object_faces:
            print(f"  skipping object {o.name}: only {len(o.faces)} faces (tapped twice?)")
    objects = [o for o in objects if len(o.faces) >= min_object_faces]

    coverage = {}
    if texture and bundle.keyframes:
        print(f"Coloring from {len(bundle.keyframes)} keyframes (depth confidence >= {min_confidence})")
        coverage = colorize([ground_visual, background, *objects], bundle.keyframes, min_confidence=min_confidence)

    print("Building collision")
    obj_specs = []
    for o in objects:
        pieces, method = object_collision(o, hf, method=collision)
        origin = body_origin(o, hf)
        info = []
        for i, (v, f) in enumerate(pieces):
            write_obj(sim / "collision" / f"{o.name}_{i}.obj", v, f)
            info.append({"file": f"collision/{o.name}_{i}.obj", "volume": hull_volume(v, f)})
        obj_specs.append({"name": o.name, "origin": origin, "mass": masses.get(o.name, default_mass),
                          "visual": o, "pieces": pieces, "piece_info": info, "method": method})

    print("Writing USD")
    usd_hf = hf.downsample(2) if hf.z.size > 250_000 else hf
    write_usd.write(sim / "scene.usda", ground_visual, usd_hf.to_mesh(), background, obj_specs, friction=friction)
    print("Writing MJCF")
    write_mjcf.write(sim, hf, ground_visual, background, obj_specs, friction=friction)

    report = {
        "bundle": bundle.root.name,
        "heightfield": {"x0": hf.x0, "y0": hf.y0, "cell": hf.cell, "shape": list(hf.z.shape),
                        "measured_fraction": round(float(hf.measured.mean()), 4),
                        "z_range": [round(float(hf.z.min()), 4), round(float(hf.z.max()), 4)],
                        "file": "ground.hfield.bin"},
        "usd_ground_cell": usd_hf.cell,
        "ground_faces": int(len(ground_visual.faces)),
        "background_faces": int(len(background.faces)),
        "objects": [{"name": s["name"], "mass": s["mass"], "origin": [round(float(x), 4) for x in s["origin"]],
                     "faces": int(len(s["visual"].faces)), "collision": s["method"], "pieces": s["piece_info"]}
                    for s in obj_specs],
        "corridor_m": corridor,
        "loop_drift": drift,
        "texture_coverage": {k: round(v, 3) for k, v in coverage.items()},
        "outputs": ["scene.usda", "scene.xml", "ground.hfield.bin", "collision/", "meshes/"],
    }
    (sim / "report.json").write_text(json.dumps(report, indent=2))
    print(f"Done: {sim}")
    return report

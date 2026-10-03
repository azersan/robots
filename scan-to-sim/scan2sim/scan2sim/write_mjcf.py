"""MJCF fallback for plain MuJoCo: heightfield ground, free-jointed objects from convex pieces.

The background (walls, house, trees) is visual-only here: MuJoCo collides with the convex hull of
a mesh, which would fill a room. Use the USD scene when background collisions matter.
"""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

from .collision import write_obj
from .ground import Heightfield


def write_hfield_bin(path: Path, hf: Heightfield):
    """MuJoCo binary hfield: int32 nrow, int32 ncol, float32 data (row 0 = lowest y)."""
    ny, nx = hf.z.shape
    with open(path, "wb") as out:
        np.array([ny, nx], dtype="<i4").tofile(out)
        hf.z.astype("<f4").tofile(out)


def write(sim_dir: Path, hf: Heightfield, ground_visual, background, objects: list[dict], friction: float = 0.8) -> Path:
    meshes = sim_dir / "meshes"
    meshes.mkdir(exist_ok=True)
    write_hfield_bin(sim_dir / "ground.hfield.bin", hf)

    ny, nx = hf.z.shape
    zmin, zmax = float(hf.z.min()), float(hf.z.max())
    rx, ry = (nx - 1) * hf.cell / 2, (ny - 1) * hf.cell / 2
    cx, cy = hf.x0 + rx, hf.y0 + ry
    zrange = max(zmax - zmin, 1e-3)

    assets = [f'<hfield name="ground" file="ground.hfield.bin" size="{rx:.4f} {ry:.4f} {zrange:.4f} 0.1"/>']
    world = [
        '<light pos="0 0 10" dir="0 0 -1" directional="true"/>',
        f'<geom name="ground" type="hfield" hfield="ground" pos="{cx:.4f} {cy:.4f} {zmin:.4f}" '
        f'friction="{friction} 0.005 0.0001" rgba="0.45 0.6 0.4 1" group="2"/>',
    ]
    for name, part, rgba in [("ground_visual", ground_visual, "0.45 0.6 0.4 1"), ("background", background, "0.6 0.6 0.6 1")]:
        if part is not None and len(part.faces):
            write_obj(meshes / f"{name}.obj", part.vertices, part.faces)
            assets.append(f'<mesh name="{name}" file="meshes/{name}.obj" inertia="shell"/>')
            world.append(f'<geom name="{name}" type="mesh" mesh="{name}" contype="0" conaffinity="0" group="1" rgba="{rgba}"/>')

    for obj in objects:
        name, origin = obj["name"], obj["origin"]
        vols = [max(p["volume"], 1e-6) for p in obj["piece_info"]]
        geoms = []
        for i, ((v, f), vol) in enumerate(zip(obj["pieces"], vols)):
            mname = f"{name}_c{i}"
            write_obj(meshes / f"{mname}.obj", v - origin, f)
            assets.append(f'<mesh name="{mname}" file="meshes/{mname}.obj"/>')
            mass = obj["mass"] * vol / sum(vols)
            geoms.append(f'<geom type="mesh" mesh="{mname}" mass="{mass:.4f}" friction="{friction} 0.005 0.0001" '
                         f'rgba="0.95 0.55 0.15 1"/>')
        world.append(f'<body name={quoteattr(name)} pos="{origin[0]:.4f} {origin[1]:.4f} {origin[2]:.4f}">'
                     f'<freejoint/>{"".join(geoms)}</body>')

    xml = ['<mujoco model="scan">',
           '  <compiler angle="radian"/>',
           '  <option gravity="0 0 -9.81" timestep="0.002"/>',
           '  <visual><global offwidth="1280" offheight="960"/></visual>',
           '  <asset>', *[f"    {a}" for a in assets], '  </asset>',
           '  <worldbody>', *[f"    {w}" for w in world], '  </worldbody>',
           '</mujoco>']
    path = sim_dir / "scene.xml"
    path.write_text("\n".join(xml) + "\n")
    return path
